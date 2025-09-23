import asyncio
import numpy as np
import plotly.graph_objects as go
from tqdm.auto import tqdm


async def acquire_data(program, soc, py_avg, queue: asyncio.Queue,
                       stop_event: asyncio.Event, latest_holder: dict, mode: str):
    """
    Acquire IQ data (1D, 2D, or decimated) and update running averages.

    Parameters
    ----------
    program : object
        QICK program object (with acquire/acquire_decimated).
    soc : object
        QICK SoC object.
    py_avg : int
        Number of averages to run.
    queue : asyncio.Queue
        Async queue for transferring data to the plot task.
    stop_event : asyncio.Event
        Event flag to signal termination.
    latest_holder : dict
        Shared dict to hold latest results (iq_avg, z_plot, count).
    mode : str
        Acquisition mode: "decimate", "1D", or "2D".
    """
    iq_sum = 0
    i = -1

    # Select acquisition function
    acquire_fn = program.acquire_decimated if mode == 'decimate' else program.acquire

    try:
        for i in tqdm(range(py_avg), desc="average count"):
            # Get IQ data
            iq = acquire_fn(soc, rounds=1, progress=False)

            # Shape depends on mode
            if mode == 'decimate':
                iq_complex = iq[0].dot([1, 1j])
            else:
                iq_complex = iq[0][0].dot([1, 1j])

            # Running average
            iq_sum += iq_complex
            iq_avg = iq_sum / (i + 1)
            latest_holder["iq_avg"] = iq_avg
            latest_holder["count"] = i + 1

            # Compute data for plotting
            if mode in ['decimate', '1D']:
                z_plot = np.abs(iq_avg)
            elif mode == '2D':
                data = np.abs(iq_avg)
                # Row-wise normalization
                z_plot = np.array([
                    (row - row.min()) / (row.max() - row.min())
                    if row.max() != row.min() else np.zeros_like(row)
                    for row in data
                ])
            else:
                raise ValueError(f"Unknown mode: {mode}")

            latest_holder["z_plot"] = z_plot

            # Keep only latest result in queue
            if queue.full():
                _ = queue.get_nowait()
            await queue.put((i + 1, z_plot))

            await asyncio.sleep(0)

    except Exception as e:
        print(f"[acquire_data] Instrument stopped: {e}")

    finally:
        latest_holder["count"] = i + 1 if i >= 0 else 0
        stop_event.set()
        while not queue.empty():
            _ = queue.get_nowait()
        await queue.put(None)


async def plot_data(figw, py_avg, title, queue: asyncio.Queue,
                    stop_event: asyncio.Event, latest_holder: dict, update_interval=0.5):
    """
    Update the plot in real-time using data from the async queue.

    Parameters
    ----------
    figw : go.FigureWidget
        Plotly figure widget to update.
    py_avg : int
        Total number of averages.
    queue : asyncio.Queue
        Queue for new data.
    stop_event : asyncio.Event
        Event flag to signal termination.
    latest_holder : dict
        Dict storing latest results.
    update_interval : int
        Update the plot every N iterations.
    """
    while not stop_event.is_set():
        if queue.empty():
            await asyncio.sleep(0.05)
            continue

        item = await queue.get()
        if item is None:
            break

        i, z_plot = item
        if (i % update_interval == 0) or (i == py_avg):
            with figw.batch_update():
                # For 1D: update y
                # For 2D: update z (will be handled by caller)
                if figw.data[0].type == "heatmap":
                    figw.data[0].z = z_plot
                else:
                    figw.data[0].y = z_plot
                figw.layout.title.text = f"{title}"
            latest_holder["z_plot"] = z_plot


async def asyn_run(program, soc, py_avg, figw, title: str, mode="decimate"):
    """
    Run acquisition and plotting loop for 1D / 2D / decimate modes.

    Parameters
    ----------
    program : object
        QICK program object.
    soc : object
        QICK SoC object.
    py_avg : int
        Number of averages.
    figw : go.FigureWidget
        Plotly figure widget for live update.
    title : str
        Final plot title.
    mode : str
        "decimate", "1D", or "2D".

    Returns
    -------
    iq_avg : np.ndarray
        Final averaged IQ data.
    z_final : np.ndarray or None
        Final normalized amplitude map (only for 2D).
    """
    queue = asyncio.Queue(maxsize=1)
    stop_event = asyncio.Event()
    latest_holder = {"iq_avg": None, "z_plot": None, "count": 0}

    # Start acquisition and plotting
    task1 = asyncio.create_task(acquire_data(program, soc, py_avg, queue, stop_event, latest_holder, mode=mode))
    task2 = asyncio.create_task(plot_data(figw, py_avg, title, queue, stop_event, latest_holder))

    try:
        await asyncio.gather(task1, task2)
    except asyncio.CancelledError:
        print(f"⚠️ Measurement interrupted at {latest_holder.get('count', 0)} averages")
    finally:
        # Freeze final result
        if latest_holder["z_plot"] is not None:
            with figw.batch_update():
                if mode == "2D":
                    figw.data[0].z = latest_holder["z_plot"]
                else:
                    figw.data[0].y = latest_holder["z_plot"]
                figw.layout.title.text = f"{title}"

        if mode == "2D":
            return latest_holder["iq_avg"], latest_holder["z_plot"]
        else:
            return latest_holder["iq_avg"]
