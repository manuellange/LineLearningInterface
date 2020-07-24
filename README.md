# LineLearningInterface

## ImageMatcher

The goal of the `ImageMatcher` is to read two images (left and right/second image) and their corresponding keylines and descriptors (from the Network) to match their descriptors. The result of the binary descriptor matching is shown as a single image with visualized keylines (and their matches).

We used `Python (3.6.9)` and `opencv-contrib-python (4.0.1.24)`.

You can run the application as follows (`.npz` files are automatically derived from image file paths):

```powershell
python3 ./imagematcher.py -l "../LineDataset/corridor/f7_rotation/rotation_left.jpg" -r "../LineDataset/corridor/f7_rotation/rotation_right.jpg" --max_dist 5000
```

You can also manually specify the `.npz` for each image as follows:

```powershell
python3 ./imagematcher.py --image_left "../LineDataset/corridor/f7_rotation/rotation_left.jpg" --npz_left "../LineDataset/corridor/f7_rotation/rotation_left.npz" --image_right "../LineDataset/corridor/f7_rotation/rotation_right.jpg" --npz_right "../LineDataset/corridor/f7_rotation/rotation_right.npz" --max_dist 5000
```
