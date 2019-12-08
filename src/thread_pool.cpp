#include "thread_pool.h"

void ThreadPool::start(size_t nthreads)
{
    if (threads.size() > 0)
        throw std::runtime_error("start() called on already started ThreadPool");

    if (nthreads == 0)
        nthreads = std::thread::hardware_concurrency();

    for (size_t i = 0; i < nthreads; ++i)
    {
        // Start worker threads and store the std::thread objects in threads queue
        threads.emplace_back(
            [this]() // this lambda is the thread callback
            {
                for (;;)
                {
                    std::function<void()> task;

                    // A new scope to contain the unique_lock
                    {
                        std::unique_lock<std::mutex> lock(this->pool_mutex);

                        // Wait for something interesting to happen
                        this->wake_worker.wait(lock,
                                               [this] { return this->stopping || !this->tasks.empty(); });

                        // If stopping, exit
                        if (this->stopping && this->tasks.empty())
                            return;

                        // Otherwise, we must have a new task.
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    task();
                }
            });
    }
}

void ThreadPool::wait()
{
    std::unique_lock<std::mutex> lock(this->pool_mutex);
    while (!statuses.empty())
    {
        while (!statuses.front()->done)
            statuses.front()->changed.wait(lock);
        statuses.pop();
    }
}

void ThreadPool::stop()
{
    stopping = true;

    wake_worker.notify_all();

    for (std::thread &t : threads)
        t.join();

    threads.clear();

    stopping = false;
}
