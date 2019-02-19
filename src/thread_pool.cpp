#include "thread_pool.h"

void ThreadPool::start(size_t nthreads)
{
    if (threads.size() > 0)
        throw std::runtime_error("start() called on already started ThreadPool");

    if (nthreads == 0)
        nthreads = std::thread::hardware_concurrency();

    for (size_t i = 0; i < nthreads; ++i)
        threads.emplace_back( // start a new thread and store the std::thread object in threads
            [this]() // this lambda is the thread callback
    {
        for (;;)
        {
            std::function<void()> task;

            // The unique_lock only exists in this scope and is unlocked at the end
            {
                std::unique_lock<std::mutex> lock(this->tasks_mutex);
                this->condition.wait(lock,
                    [this] { return this->stopping || !this->tasks.empty(); });
                if (this->stopping && this->tasks.empty())
                    return;
                task = std::move(this->tasks.front());
                this->tasks.pop();
            }

            task();
        }
    }
    );
}

void ThreadPool::wait()
{
    std::unique_lock<std::mutex> lock(this->dones_mutex);
    while (!dones.empty())
    {
        while (!dones.front()->done)
            dones.front()->wake.wait(lock);
        dones.pop();
    }
}

void ThreadPool::stop()
{
    stopping = true;

    condition.notify_all();

    for (std::thread &t : threads)
        t.join();

    threads.clear();

    stopping = false;
}
