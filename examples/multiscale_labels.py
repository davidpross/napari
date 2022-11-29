"""
Multiscale Labels 2D
=========

Create a multiscale labels layer to test performance with direct coloring

"""

from skimage import data
from skimage.transform import resize
import dask.array as da
from scipy import ndimage as ndi
import napari
import tempfile

def labeled_blobs(arr):
    blobs = data.binary_blobs(length=1024, volume_fraction=0.1, n_dim=2)
    return ndi.label(blobs)[0]

def _downsample(image):
    return resize(
        image,
        output_shape=(image.shape[0] // 2, image.shape[1] // 2),
        order=0,
        preserve_range=True,
        anti_aliasing=False
    )

arr = da.zeros((102400, 102400),
    dtype='uint16',
    chunks=(1024, 1024)).map_blocks(
        labeled_blobs,
        dtype='uint16'
    )
LEVELS = 7

with tempfile.TemporaryDirectory() as tmpdirname:
    print('created temporary directory', tmpdirname)
    for i in range(LEVELS):
        print(f"Writing level {i}, shape: ({arr.shape[0]}, {arr.shape[1]})")
        arr.to_zarr(tmpdirname, component=f"{i}", overwrite=True)
        new_chunks = tuple([
            tuple([i // 2 for i in arr.chunks[0]]),
            tuple([i // 2 for i in arr.chunks[1]])
        ])
        arr = arr.map_blocks(_downsample, dtype='uint16', chunks=new_chunks)
    
    labels = [da.from_zarr(tmpdirname, component=f"{i}") for i in range(LEVELS)]
    # add the labels
    napari.view_labels(labels)

    if __name__ == '__main__':
        napari.run()
