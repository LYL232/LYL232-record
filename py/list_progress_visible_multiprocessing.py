"""
带有进度条的多进程执行器，使用方法见main，只能用于multiprocessing的fork启动方式
"""
import tqdm
from multiprocessing import Queue, Condition, Process
from typing import Callable
from threading import Thread


class ListProgressVisibleMultiprocessing:
    """
    生产者消费者模型下的带进度条的多进程任务执行器
    """

    def __init__(self, processes: int, process_func: Callable):
        self.__processes = processes
        self.__process_func = process_func

    def process(
            self, todo_list: list, batch, verbose: bool = True,
            kwargs: dict = None
    ):
        kwargs = {} if kwargs is None else kwargs
        has_task = Condition()
        no_task = Condition()
        task_queue = Queue()
        result_queue = Queue()

        processes = []
        for i in range(self.__processes):
            target_kwargs = kwargs.copy()
            target_kwargs.update({
                'process_id': i,
                'task_queue': task_queue,
                'has_task': has_task,
                'no_task': no_task,
                'result_queue': result_queue,
            })
            processes.append(
                Process(target=self._process_main, kwargs=target_kwargs)
            )

        for each in processes:
            each.start()

        task_send_thread = Thread(target=self._send_task_thread_main, kwargs={
            'task_list': todo_list,
            'task_queue': task_queue,
            'has_task': has_task,
            'no_task': no_task,
            'batch': batch
        })
        task_send_thread.start()

        result_list_dict = {}
        with tqdm.tqdm(range(len(todo_list)), disable=not verbose) as bar:
            while len(result_list_dict) < len(todo_list):
                completed = result_queue.get()
                for index, content in completed:
                    result_list_dict[index] = content
                bar.update(len(completed))

        for index, each in enumerate(processes):
            each.join()
        task_send_thread.join()
        result = []
        for i in range(len(todo_list)):
            result.append(result_list_dict[i])
        return result

    def _send_task_thread_main(
            self, task_list: list,
            task_queue: Queue,
            has_task: Condition,
            no_task: Condition,
            batch: int
    ):
        done = False
        begin = 0
        put_tasks_each = self.__processes + 1
        end = min(put_tasks_each * batch, len(task_list))
        while not done:
            putting_batches = []
            batched_tasks = []
            i = begin
            while len(putting_batches) < put_tasks_each and i < end:
                if len(batched_tasks) >= batch:
                    putting_batches.append(batched_tasks)
                    batched_tasks = []
                batched_tasks.append((i, task_list[i]))
                i += 1
            if len(batched_tasks) > 0:
                putting_batches.append(batched_tasks)
            with no_task:
                while task_queue.qsize() >= self.__processes + 1:
                    no_task.wait()
                for each in putting_batches:
                    task_queue.put(each)

                if end == len(task_list):
                    done = True
                    task_queue.put([])

            with has_task:
                has_task.notify_all()
            begin = end
            end = min(end + put_tasks_each * batch, len(task_list))

    def _process_main(
            self,
            process_id: int,
            task_queue: Queue,
            has_task: Condition,
            no_task: Condition,
            result_queue: Queue,
            **kwargs
    ):
        kwargs['process_id'] = process_id
        while True:
            with has_task:
                while task_queue.qsize() <= 0:
                    with no_task:
                        no_task.notify_all()
                    has_task.wait()
                batch_inputs = task_queue.get()
            if len(batch_inputs) == 0:
                task_queue.put([])
                with has_task:
                    has_task.notify_all()
                break
            result_list = []
            for index, target in batch_inputs:
                result_list.append(
                    (index, self.__process_func(target, **kwargs))
                )
            if len(result_list) > 0:
                result_queue.put(result_list)


def main():
    def process_func(task, **kwargs):
        """
        每个进程对每个传入对象的处理函数
        :param task: 下文中todo_list内的对象
        :param kwargs: 额外参数，可通过process的kwargs字典传入，
        会额外传入一个参数：process_id，范围从0到进程减1
        :return: 不返回值
        """
        # 解除下列注释可以看到各个进程交替执行不同的Task对象
        # print('Task', task, kwargs)

    processor = ListProgressVisibleMultiprocessing(
        4, process_func=process_func
    )

    todo_list = [i for i in range(1000000)]

    processor.process(todo_list, batch=1)


if __name__ == '__main__':
    main()
