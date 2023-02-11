import logging
import queue

import av
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from transformers import pipeline

logger = logging.getLogger(__name__)


result_queue: queue.Queue = (
    queue.Queue()
)  # TODO: A general-purpose shared state object may be more useful.


@st.cache_resource
def get_pipeline():
    return pipeline("object-detection", model="hustvl/yolos-small")


obj_detector = get_pipeline()


def callback(frame: av.VideoFrame) -> av.VideoFrame:
    image_ndarray = frame.to_ndarray(format="bgr24")

    image = Image.fromarray(image_ndarray)

    results = obj_detector(image)

    # Ref: https://huggingface.co/docs/transformers/main/tasks/object_detection#inference  # noqa: E501
    draw = ImageDraw.Draw(image)
    for result in results:
        box = result["box"]
        draw.rectangle(
            (box["xmin"], box["ymin"], box["xmax"], box["ymax"]), outline="red", width=1
        )
        draw.text((box["xmin"], box["ymin"]), result["label"], fill="white")

    # NOTE: This callback is called in another thread,
    # so it must be thread-safe.
    result_queue.put(results)

    return av.VideoFrame.from_ndarray(np.array(image), format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if st.checkbox("Show the detected labels", value=True):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        # NOTE: The video transformation with object detection and
        # this loop displaying the result labels are running
        # in different threads asynchronously.
        # Then the rendered video frames and the labels displayed here
        # are not strictly synchronized.
        while True:
            result = result_queue.get()
            labels_placeholder.table(result)

st.markdown(
    "This demo uses a model and code from "
    "https://github.com/robmarkcole/object-detection-app. "
    "Many thanks to the project."
)
