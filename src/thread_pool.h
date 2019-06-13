#pragma once

#ifndef GUARD_thread_pool_h
#define GUARD_thread_pool_h

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <memory>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

// Private class ThreadPool uses to track task status
class ThreadPoolTaskStatus
{
    friend class ThreadPool;

  private:
    inline ThreadPoolTaskStatus() : done(false){};
    std::atomic<bool> done;
    std::condition_variable changed;

    // No copies allowed because conditional_variable can't be copied
    ThreadPoolTaskStatus(const ThreadPoolTaskStatus &) = delete;
};

class ThreadPool
{
  public:
    inline ThreadPool() : stopping(false){};
    inline ~ThreadPool() { stop(); }

    // Start the worker threads
    // If nthreads = 0, start 1 thread per hardware core
    // This should only be called once, unless stop is called in between
    void start(size_t nthreads = 0);

    // Stop the worker threads, and wait for them to end.
    // If this isn't called manually, it's called automatically by the destructor
    void stop();

    // Schedule a task for a worker to pick up.
    // Returns a future that will provide the return value, if any.
    // This must be in the header because it's a template function.
    template <class F, class... Args>
    auto add_task(F &&f, Args &&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>
    {
        if (threads.size() == 0)
            throw std::runtime_error("add_task() called on inactive ThreadPool");

        if (stopping)
            throw std::runtime_error("add_task() called on stopping ThreadPool");

        // Get return type from passed function f
        using return_type = typename std::result_of<F(Args...)>::type;

        // Created a shared_ptr to a packaged_task for f(args)
        auto sharedf = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        // Add a new status, shared by the closure below and the queue for wait()
        auto status = new ThreadPoolTaskStatus;
        status_mutex.lock();
        statuses.push(std::shared_ptr<ThreadPoolTaskStatus>(status));
        status_mutex.unlock();

        // Get a future to pass back the return value from f
        std::future<return_type> res = sharedf->get_future();

        // When this lambda is called by a worker, it will call f(args),
        // then set the done flag in the status.
        tasks_mutex.lock();
        tasks.emplace(
            [this, sharedf, status]() {
                (*sharedf)();
                status_mutex.lock();
                status->done = true;
                status->changed.notify_all();
                status_mutex.unlock();
            });
        tasks_mutex.unlock();

        // Wake a worker to perform the task
        wake_worker.notify_one();

        // Return the future
        return res;
    }

    // Wait for all the tasks to complete.
    void wait();

    // Are the worker threads running?
    inline bool is_active() { return !stopping && threads.size() > 0; }

  private:
    std::vector<std::thread> threads;
    std::queue<std::shared_ptr<ThreadPoolTaskStatus>> statuses;
    std::queue<std::function<void()>> tasks;

    // synchronization
    std::mutex tasks_mutex;
    std::mutex status_mutex;
    std::condition_variable wake_worker;
    std::atomic<bool> stopping;
};

#endif
