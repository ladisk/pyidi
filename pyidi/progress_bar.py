import rich.progress

def progress_bar(start, stop, step, show_pbar=True):
    """
    Create a progress bar range or a normal range.
    """
    if show_pbar:
        def generator():
            with rich_progress_bar_setup() as progress:
                task = progress.add_task("Processing", total=(stop-start)//step)
                for i in range(start, stop, step):
                    progress.update(task, advance=1)
                    yield i

        return generator()
    else:
        return range(start, stop, step)
    
def rich_progress_bar_setup():
    """Configure the rich progress bar."""
    return rich.progress.Progress(
                "[progress.description]{task.description}",
                rich.progress.MofNCompleteColumn(),
                rich.progress.BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                rich.progress.TimeRemainingColumn(),
                rich.progress.TimeElapsedColumn(),
                refresh_per_second=5,
            )