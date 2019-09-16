import math
import copy

import pandas as pd
import matplotlib.pyplot as plt


def load_task(f, names):
    df = pd.read_csv(f, header=0, names=names)

    df['end'] = pd.to_numeric(df['end'])
    df['end'] = df['end'] / 1000
    df['start'] = pd.to_numeric(df['start'])
    df['start'] = df['start'] / 1000
    df['Diff'] = df.end - df.start
    df.sort_values('start', inplace=True)

    # montage colors
    color = {'mDiffFit': 'turquoise', 'mProjectPP': 'crimson', 'mConcatFit': 'green', 'mBackground': 'blue',
             'mJPEG': 'yellow', 'mShrink': 'brown', 'mAdd': 'orange', 'mBgModel': 'black', 'mImgtbl': 'violet',
             'kinc-wrapper': 'red'}

    return df, color


def load_diagnostic(f=None, names=None):
    df = pd.read_csv(f, header=0, names=names)

    df['time'] = pd.to_numeric(df['time'])
    df['time'] = df['time'] / 1000 / 1000 / 1000
    df.sort_values('time', inplace=True)

    return df


def load_performance(f=None, names=None):
    df = pd.read_csv(f, header=0, names=names)

    return df


def gannt(f=None, names=None, df=None, color=None, show=True, limit=None):
    plt.clf()

    if df is None:
        df, color = load_task(f, names)
    else:
        df = copy.copy(df)

    df['start'] = df['start'] - df['start'][0]

    fig, ax = plt.subplots(figsize=(6, 3))

    labels = []
    i = 0

    for x in df.iterrows():
        labels.append(x[1].taskID)

        ax.broken_barh([(x[1].start, x[1].Diff)], (i - 0.1, 0.8), color=color[x[1].taskID])

        i += 1

        if i == limit:
            break

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Czas [s]")
    ax.set_xlim(left=0)
    plt.title('Wykonanie workflowu')
    plt.tight_layout()

    if show:
        plt.show()
    else:
        return plt


def gannt_traces(f=None, names=None, df=None, show=True, limit=None):
    plt.clf()

    if df is None:
        df, color = load_task(f, names)
    else:
        df = copy.copy(df)

    delta = df['download_start'][0]

    for stage in ('download', 'execute', 'upload'):
        for metric in (stage + '_start', stage + '_end'):
            df[metric] = df[metric] - delta

    vertical_positions = {task: index for index, task in enumerate(df.taskID.unique())}
    legend = ('download', 'execute', 'upload')

    fig, ax = plt.subplots(figsize=(6, 3))

    labels = []
    i = 0

    for x in df.iterrows():
        task = x[1].taskID
        labels.append(task)
        yrange = (vertical_positions[task] - 0.4, 0.8)

        a1 = ax.broken_barh([(x[1].download_start / 1e3, x[1].download_end / 1e3 - x[1].download_start / 1e3)], yrange,
                            color='red')
        a2 = ax.broken_barh([(x[1].execute_start / 1e3, x[1].execute_end / 1e3 - x[1].execute_start / 1e3)], yrange,
                            color='green')
        a3 = ax.broken_barh([(x[1].upload_start / 1e3, x[1].upload_end / 1e3 - x[1].upload_start / 1e3)], yrange,
                            color='blue')

        ax.legend([a1, a2, a3], legend)

        i += 1

        if i == limit:
            break

    ax.set_yticks(range(len(vertical_positions)))
    ax.set_yticklabels(vertical_positions)
    ax.set_xlabel("Czas [s]")
    ax.set_xlim(left=0)

    plt.title('Wykonanie workflowu z podziałem na etapy')
    plt.tight_layout()

    if show:
        plt.show()
    else:
        return plt


def mean_tasks_duration(f=None, names=None, df=None, show=True):
    plt.clf()

    if df is None:
        df, color = load_task(f, names)
    else:
        df = copy.deepcopy(df)

    df['start'] = df['start'] - df['start'][0]

    df['diff_download'] = (df['download_end'] - df['download_start']) / 1000
    df['diff_upload'] = (df['upload_end'] - df['upload_start']) / 1000
    df['diff_exec'] = (df['execute_end'] - df['execute_start']) / 1000

    fig, ax = plt.subplots()

    taskIDs = df.taskID.unique()
    y_pos = list(range(len(taskIDs)))
    width = 0.3

    performance_download = {task: df.loc[df.taskID == task].diff_download.mean() for task in taskIDs}
    performance_upload = {task: df.loc[df.taskID == task].diff_upload.mean() for task in taskIDs}
    performance_exec = {task: df.loc[df.taskID == task].diff_exec.mean() for task in taskIDs}

    for dataset, w, label in ((performance_download, -1, 'download'), (performance_exec, 1, 'execute'),
                              (performance_upload, 3, 'upload')):
        data = sorted(dataset.items(), key=lambda x: x[0])

        ax.barh(list(map(lambda x: x + w * width / 2, y_pos)), list(map(lambda x: x[1], data)), width, align='center',
                label=label)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(map(lambda x: x[0], data)))

    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Czas [s]')
    ax.set_title('Średni czas wykonania zadań - Montage 2.0')
    ax.legend()

    if show:
        plt.show()
    else:
        return plt


def box_mean(f=None, names=None, df=None, show=True):
    plt.clf()

    if df is None:
        df, color = load_task(f, names)
    else:
        df = copy.deepcopy(df)

    df['start'] = df['start'] - df['start'][0]

    allTasks = df.taskID.unique()

    y_length = len(allTasks) // 2 + 1
    fig, axs = plt.subplots(2, y_length)

    for index, task in enumerate(allTasks):
        x, y = index // y_length, index % y_length

        data = df.loc[df.taskID == task].Diff

        ax = axs[x, y]

        ax.boxplot(data)

        ax.set_title(task)
        ax.set_ylabel('Czas wykonania [s]')
        ax.set_xticklabels([])

    if len(allTasks) % 2 != 0:
        fig.delaxes(axs[-1, -1])

    fig.subplots_adjust(wspace=0.3)
    fig.suptitle('Czasy wykonania zadań')

    if show:
        plt.show()
    else:
        return fig


def concurrent_tasks(f=None, names=None, df=None, show=True):
    plt.clf()

    if df is None:
        df, color = load_task(f, names)

    window = 50  # seconds
    N = 0

    # cut out last 4 tasks
    df = df[:-4]
    #

    start = df.iloc[0].start
    end = df.iloc[-1].start

    data = []

    while start < end:
        _end = start + window

        concurrent = len(df[((df.start >= start) & (df.start <= _end)) | ((df.end >= start) & (df.end <= _end)) |
                            ((df.start <= start) & (df.end >= _end))])
        data.append(concurrent)

        start += window
        N += 1

    ind = list(range(N))  # the x locations for the groups

    plt.bar(ind, data)

    plt.ylabel('Ilość zadań')
    plt.title('Równolegle wykonywane zadania')

    plt.xticks(range(0, N, 20), range(0, int(end - df.iloc[0].start), window * 20))
    plt.xlabel('Czas [s]')

    if show:
        plt.show()
    else:
        return plt


def avg_perf(f=None, names=None, df=None, show=True):
    plt.clf()

    if df is None:
        df = load_performance(f, names)

    taskIDs = df.taskID.unique()

    for metric in ['cpu_usage', 'mem_usage', 'conn_recv']:
        data = [df[(df.taskID == task) & (df[metric] != '') & (df[metric] != '-1')][metric].mean() for task in taskIDs]
        ind = list(range(len(taskIDs)))

        plt.bar(ind, data)

        plt.ylabel('{} usage'.format(metric))
        plt.xlabel('Task type')
        plt.title('Mean {} per task type'.format(metric))

        plt.xticks(range(0, len(taskIDs)), taskIDs)

        if show:
            plt.show()
        else:
            yield plt


def concurrent_tasks_with_waiting(f=None, names=None, f1=None, names1=None, df=None, df1=None, show=True):
    plt.clf()

    if df is None:
        df, color = load_task(f, names)
        df1 = load_diagnostic(f1, names1)

    window = 50  # seconds
    N = 0

    start = df.iloc[0].start
    end = df.iloc[-1].start

    executed = []
    waiting = []

    while start < end:
        _end = start + window

        concurrent = len(df[((df.start >= start) & (df.start <= _end)) | ((df.end >= start) & (df.end <= _end)) |
                            ((df.start <= start) & (df.end >= _end))])

        try:
            avg_waiting_tasks = df1[(df1.time >= start) & (df1.time <= _end)].waitingTasks.mean()
        except:
            avg_waiting_tasks = 0

        executed.append(concurrent)
        waiting.append(-avg_waiting_tasks)

        start += window
        N += 1

    ind = list(range(N))  # the x locations for the groups

    p1 = plt.bar(ind, executed)
    p2 = plt.bar(ind, waiting)
    plt.legend((p1[0], p2[0]), ('Wykonywane', 'Oczekujące'))

    plt.ylabel('Ilość zadań')
    plt.title('Wykonywane i oczekujące zadania w {}-sekundowych interwałach'.format(window))

    plt.xticks(range(0, N, 2), range(0, int(end - df.iloc[0].start), window * 2))

    # todo: fix y labels
    # plt.yticks([str(abs(int(x.get_text() or 0))) for x in plt.yticks()[1]])

    plt.xlabel('Czas [s]')

    if show:
        plt.show()
    else:
        return plt


def grid_perf(f=None, names=None, df=None, show=True):
    plt.clf()

    if df is None:
        df = load_performance(f, names)

    taskIDs = df.taskID.unique()
    metrics = ['cpu_usage', 'mem_usage', 'conn_recv', 'conn_transferred', 'disk_read', 'disk_write']

    columns = len(metrics)
    rows = len(taskIDs)
    index = 1

    for task in taskIDs:
        for metric in metrics:
            plt.subplot(columns, rows, index)

            # add title to left most column
            if (index - 1 + columns) % columns == 0:
                plt.ylabel(task)

            # add title only to top most row
            if index <= columns:
                plt.title(metric)

            data = df[(df.taskID == task) & (df[metric] != '') & (df[metric] != '-1')][metric].mean()

            index += 1

            plt.bar([0], [data])

    plt.subplots_adjust(hspace=0.6, wspace=0.3)

    if show:
        plt.show()
    else:
        return plt


def grid_perf_all(f=None, names=None, experiment=None, df=None, show=True):
    plt.clf()

    if df is None:
        df = load_performance(f, names)

    taskIDs = df.taskID.unique()

    metrics = ['cpu_usage', 'mem_usage', 'conn_recv', 'conn_transferred', 'disk_read', 'disk_write']

    aa = df[(df.experiment == experiment)] if experiment else df

    c = {'mProjectPP': 307, 'mDiffFit': 862}

    for task in taskIDs:
        samples = aa[(aa.taskID == task)]
        samples_count = min(len(samples), 100)
        columns = rows = math.ceil(math.sqrt(samples_count))
        ids = samples.containerID.unique()[:c[task]]

        for metric in metrics:

            print('==> Creating grids of {} graphs ({} for each grid ) for metric {}'.format(len(ids), samples_count,
                                                                                             metric))

            for offset in range(0, len(ids), samples_count):
                fig = plt.figure(figsize=(15, 15))
                for index, containerId in enumerate(ids[offset:offset + samples_count]):
                    container_samples = samples[(samples.containerID == containerId) & (samples[metric] != '-1') &
                                                (samples[metric] != '') & (samples[metric].notnull())]
                    container_samples.time = (container_samples.time - container_samples.time.min()) / 1e9

                    ax = fig.add_subplot(columns, rows, index + 1)
                    ax.plot(container_samples.time, container_samples[metric])

                fig.suptitle(' - '.join((task, metric)))
                fig.subplots_adjust(hspace=0.6, wspace=0.7)

                if show:
                    plt.show()
                else:
                    yield fig

                print('==> {}% done'.format(round((offset + samples_count) / len(ids) * 100, 2)))


if __name__ == '__main__':
    box_mean(
        'example-dataset/hflow_task.csv',
        'name,tags,time,configId,containerID,download_end,download_start,end,execute_end,execute_start,experiment,'
        'start,taskID,upload_end,upload_start'.split(','))
