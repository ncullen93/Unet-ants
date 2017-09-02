"""
NOTE TO USER: the `DataLoader` class is the only 

"""
import multiprocessing
import numpy as np
import collections
import math
import sys
import traceback
import threading

import scipy.misc
import os

if sys.version_info[0] == 2:
    import Queue as queue
    string_classes = basestring
else:
    import queue
    string_classes = (str, bytes)


_use_shared_memory = False
"""Whether to use shared memory in default_collate"""


class ExceptionWrapper(object):
    "Wraps an exception plus traceback to communicate across threads"

    def __init__(self, exc_info):
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))


def _worker_loop(dataset, index_queue, data_queue, collate_fn):
    global _use_shared_memory
    _use_shared_memory = True

    while True:
        r = index_queue.get()
        if r is None:
            data_queue.put(None)
            break
        idx, batch_indices = r
        try:
            samples = collate_fn([dataset[i] for i in batch_indices])
        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))


def default_collate(batch):
    """Puts each data field into a tensor with outer dimension batch size"""
    if type(batch[0]).__module__ == 'numpy':
        elem = batch[0]
        if type(elem).__name__ == 'ndarray':
            return np.stack([b for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return np.array(list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return np.array(batch).astype('uint8')
    elif isinstance(batch[0], float):
        return np.array(batch).astype('float32')
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))


class DataLoaderIter(object):
    "Iterates once over the DataLoader's dataset, as specified by the sampler"

    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.collate_fn = loader.collate_fn
        self.sampler = loader.sampler
        self.num_workers = loader.num_workers
        self.done_event = threading.Event()

        self.samples_remaining = len(self.sampler)
        self.sample_iter = iter(self.sampler)

        if self.num_workers > 0:
            self.index_queue = multiprocessing.SimpleQueue()
            self.data_queue = multiprocessing.SimpleQueue()
            self.batches_outstanding = 0
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}

            self.workers = [
                multiprocessing.Process(
                    target=_worker_loop,
                    args=(self.dataset, self.index_queue, self.data_queue, self.collate_fn))
                for _ in range(self.num_workers)]

            for w in self.workers:
                w.daemon = True  # ensure that the worker exits on process exit
                w.start()

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()

    def __len__(self):
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    def __next__(self):
        if self.num_workers == 0:  # same-process loading
            if self.samples_remaining == 0:
                if self.loader.sample_forever:
                    self.__init__(self.loader)
                else:
                    raise StopIteration
            indices = self._next_indices()
            batch = self.collate_fn([self.dataset[i] for i in indices])
            return batch

        # check if the next sample has already been generated
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._process_next_batch(batch)

        if self.batches_outstanding == 0:
            if self.loader.sample_forever:
                self._shutdown_workers()
                self.__init__(self.loader)
            else:
                self._shutdown_workers()
                raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self.data_queue.get()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
            return self._process_next_batch(batch)

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self

    def _next_indices(self):
        batch_size = min(self.samples_remaining, self.batch_size)
        batch = [next(self.sample_iter) for _ in range(batch_size)]
        self.samples_remaining -= len(batch)
        return batch

    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.num_workers
        if self.samples_remaining > 0:
            self.index_queue.put((self.send_idx, self._next_indices()))
            self.batches_outstanding += 1
            self.send_idx += 1

    def _process_next_batch(self, batch):
        self.rcvd_idx += 1
        self._put_indices()
        if isinstance(batch, ExceptionWrapper):
            raise batch.exc_type(batch.exc_msg)
        return batch

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("DataLoaderIterator cannot be pickled")

    def _shutdown_workers(self):
        if not self.shutdown:
            self.shutdown = True
            self.done_event.set()
            for _ in self.workers:
                self.index_queue.put(None)

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()


class Sampler(object):
    """Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, num_samples, data_samples=None):
        self.num_samples = num_samples
        self.data_samples = data_samples if data_samples is not None else num_samples
        self.n_repeats = math.ceil(self.num_samples / self.data_samples)

    def __iter__(self):
        return iter(np.tile(np.arange(self.data_samples,dtype='uint8'),self.n_repeats))

    def __len__(self):
        return self.num_samples

class RandomSampler(Sampler):

    def __init__(self, num_samples, data_samples):
        self.num_samples = num_samples
        self.data_samples = data_samples if data_samples is not None else num_samples
        self.n_repeats = math.ceil(self.num_samples / self.data_samples)

    def __iter__(self):
        return iter(np.tile(np.random.permutation(self.data_samples).astype('uint8'),self.n_repeats))

    def __len__(self):
        return self.num_samples


class DataLoader(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, the ``shuffle`` argument is ignored.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        collate_fn (callable, optional)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, 
                 num_workers=0, sample_forever=True, collate_fn=default_collate,
                 max_epoch=500):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.sample_forever = sample_forever

        num_samples = len(dataset) #if num_workers == 0 else int(len(dataset)*max_epoch)
        if shuffle:
            self.sampler = RandomSampler(num_samples, len(dataset))
        else:
            self.sampler = SequentialSampler(num_samples, len(dataset))

    def __iter__(self):
        return DataLoaderIter(self)

    def __len__(self):
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    def write_a_batch(self, save_dir):
        myiter = iter(self)
        x, y = myiter.next()

        if not os.path.exists(save_dir):
            try:
                os.mkdir(save_dir)
            except:
                pass
        else:
            try:
                os.rmdir(save_dir)
            except:
                pass
        
        for i in range(len(x)):
            xx = x[i]
            yy = y[i]
            scipy.misc.imsave(os.path.join(save_dir,'%3i_x.jpg'%i), np.squeeze(xx))
            scipy.misc.imsave(os.path.join(save_dir,'%3i_y.jpg'%i), np.squeeze(yy))



