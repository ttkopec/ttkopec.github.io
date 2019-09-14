import os
from argparse import ArgumentParser

import graph

html_template = '''
<!DOCTYPE html>
<html>
<head>
<title>HyperFlow</title>
</head>
<body>
    {body}
</body>
</html>
'''

image_template = '''
<div>
    <h3>{title}</h3>
    <img src={path}>
</div>
'''


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('diagnostic', help='diagnostic csv file')
    parser.add_argument('performance', help='performance csv file')
    parser.add_argument('task', help='task csv file')
    parser.add_argument('--workflow', help='workflow name')

    return parser.parse_args()


def read_line(file):
    with open(file, 'r') as fp:
        return fp.readline().strip()


def get_output_dir():
    output_dir = os.path.join(os.path.dirname(__file__), 'report')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    return output_dir
import matplotlib.pyplot as plt

def save(fig, index=None):
    output_dir = get_output_dir()
    filename = '{}_{}'.format(fig._suptitle.get_text(), index) if index else fig._suptitle.get_text()
    filename = filename.replace(' ', '_')

    fig.savefig(os.path.join(output_dir, filename))

    plt.close()



    print('Saved {}'.format(filename))


def save_html():
    output_dir = get_output_dir()
    images = []

    for filename in sorted(os.listdir(output_dir)):
        if filename == 'graphs.html':
            continue

        title = os.path.splitext(filename)[0].replace('_', ' ')

        images.append(
            image_template.format(title=title, path=os.path.join('report', filename))
        )

    output_html = html_template.format(body=''.join(images))

    with open(os.path.join(output_dir, '..', 'index.html'), 'w') as fp:
        fp.write(output_html)


def main(diagnostic, performance, task):
    # load data frames
    df_task, color = graph.load_task(task, read_line(task).split(','))
    df_diagnostic = graph.load_diagnostic(diagnostic, read_line(diagnostic).split(','))
    df_performance = graph.load_performance(performance, read_line(performance).split(','))

    # generate graphs
    # save(graph.gannt(df=df_task, color=color, show=False, limit=30))
    # save(graph.gannt_traces(df=df_task, show=False, limit=30))
    # save(graph.mean_tasks_duration(df=df_task, show=False))
    # save(graph.box_mean(df=df_task, show=False))
    # save(graph.concurrent_tasks(df=df_task, show=False))
    # save(graph.concurrent_tasks_with_waiting(df=df_task, df1=df_diagnostic, show=False))
    # save(graph.grid_perf(df=df_performance, show=False))
    # for index, plt in enumerate(graph.avg_perf(df=df_performance, show=False)):
    #     save(plt, index)

    # for index, plt in enumerate(graph.grid_perf_all(df=df_performance, show=False)):
    #     save(plt, index)

    # embed graphs in html
    save_html()


if __name__ == '__main__':
    parser = get_parser()

    main(parser.diagnostic, parser.performance, parser.task)
