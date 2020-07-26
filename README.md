# LineLearningInterface

## ImageMatcher

The goal of the `ImageMatcher` is to read two images (left and right/second image) and their corresponding keylines and descriptors (from the Network) to match their descriptors. The result of the descriptor matching is shown as a single image with visualized keylines (and their matches).

If you provide Ground Truth information, you can use the argument *--count_gt* to count the number of correct and false matches.
Ground Truth information has to be provided by ensuring that the matching lines have the same index in the first/left and the second/right image.

We used `Python (3.6.9)` and `opencv-contrib-python (4.0.1.24)`.

You can run the application as follows (`.npz` files are automatically derived from image file paths):

```powershell
python3 ./imagematcher.py -l "../LineDataset/corridor/f7_rotation/im0.png" -r "../LineDataset/corridor/f7_rotation/im1.png" --max_dist 5000
```

You can also manually specify the `.npz` for each image as follows:

```powershell
python3 ./imagematcher.py --image_left "../LineDataset/corridor/f7_rotation/im0.png" --npz_left "../LineDataset/corridor/f7_rotation/im0.npz" --image_right "../LineDataset/corridor/f7_rotation/im1.png" --npz_right "../LineDataset/corridor/f7_rotation/im1.npz" --max_dist 5000
```
