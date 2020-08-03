package com.tensorspeech.tensorflowtts.utils;

import android.os.Looper;
import android.os.Process;

import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author {@link "mailto:xuefeng.ding@outlook.com" "Xuefeng Ding"}
 * Created 2020-07-20 17:25
 */
@SuppressWarnings("unused")
public class ThreadPoolManager {

    public static ThreadPoolManager getInstance() {
        return ThreadPoolManager.Holder.INSTANCE;
    }

    private static final class Holder {
        private static final ThreadPoolManager INSTANCE = new ThreadPoolManager();
    }

    private ThreadPoolExecutor mExecutor;

    /**
     * Constructor
     */
    private ThreadPoolManager() {
        int corePoolSize = Runtime.getRuntime().availableProcessors() * 2 + 1;
        ThreadFactory namedThreadFactory = new NamedThreadFactory("thread pool");

        mExecutor = new ThreadPoolExecutor(
                corePoolSize,
                corePoolSize * 10,
                1,
                TimeUnit.HOURS,
                new LinkedBlockingQueue<>(),
                namedThreadFactory,
                new ThreadPoolExecutor.DiscardPolicy()
        );
    }

    /**
     * 执行任务
     * @param runnable 需要执行的异步任务
     */
    public void execute(Runnable runnable) {
        if (runnable == null) {
            return;
        }
        mExecutor.execute(runnable);
    }

    /**
     * single thread with name
     * @param name 线程名
     * @return 线程执行器
     */
    public ScheduledThreadPoolExecutor getSingleExecutor(String name) {
        return getSingleExecutor(name, Thread.NORM_PRIORITY);
    }

    /**
     * single thread with name and priority
     * @param name thread name
     * @param priority thread priority
     * @return Thread Executor
     */
    @SuppressWarnings("WeakerAccess")
    public ScheduledThreadPoolExecutor getSingleExecutor(String name, int priority) {
        return new ScheduledThreadPoolExecutor(
                1,
                new NamedThreadFactory(name, priority));
    }

    /**
     * 从线程池中移除任务
     * @param runnable 需要移除的异步任务
     */
    public void remove(Runnable runnable) {
        if (runnable == null) {
            return;
        }
        mExecutor.remove(runnable);
    }

    /**
     * 为线程池内的每个线程命名的工厂类
     */
    private static class NamedThreadFactory implements ThreadFactory {
        private static final AtomicInteger POOL_NUMBER = new AtomicInteger(1);
        private final ThreadGroup group;
        private final AtomicInteger threadNumber = new AtomicInteger(1);
        private final String namePrefix;
        private final int priority;

        /**
         * Constructor
         * @param namePrefix 线程名前缀
         */
        private NamedThreadFactory(String namePrefix) {
            this(namePrefix, Thread.NORM_PRIORITY);
        }

        /**
         * Constructor
         * @param threadName 线程名前缀
         * @param priority 线程优先级
         */
        private NamedThreadFactory(String threadName, int priority) {
            SecurityManager s = System.getSecurityManager();
            group = (s != null) ? s.getThreadGroup() :
                    Thread.currentThread().getThreadGroup();
            namePrefix = threadName + "-" + POOL_NUMBER.getAndIncrement();
            this.priority = priority;
        }

        @Override
        public Thread newThread(Runnable r) {
            Thread t = new Thread(group, r,
                    namePrefix + threadNumber.getAndIncrement(),
                    0);
            if (t.isDaemon()) {
                t.setDaemon(false);
            }

            t.setPriority(priority);

            switch (priority) {
                case Thread.MIN_PRIORITY:
                    Process.setThreadPriority(Process.THREAD_PRIORITY_LOWEST);
                    break;
                case Thread.MAX_PRIORITY:
                    Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO);
                    break;
                default:
                    Process.setThreadPriority(Process.THREAD_PRIORITY_FOREGROUND);
                    break;
            }

            return t;
        }
    }

    /**
     * 判断当前线程是否为主线程
     * @return {@code true} if the current thread is main thread.
     */
    public static boolean isMainThread() {
        return Looper.myLooper() == Looper.getMainLooper();
    }
}