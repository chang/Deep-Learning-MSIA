# Classifier demo
import numpy as np
import matplotlib.pyplot as plt
import PIL
import tqdm
import skimage.exposure


def hide_axes(ax):
    ax.set_xticks([]), ax.set_yticks([])


class Heatmap:
    def __init__(self, model):
        self.nclasses = 10
        self.model = model

    def make_masks(self, im, n=8, maskval=0.0):
        masks = []
        xwidth, ywidth = int(np.ceil(im.shape[1] / n)), int(np.ceil(im.shape[2] / n))
        for i in range(n):
            for j in range(n):
                mask = np.ones(im.shape[1:3])
                mask[(i * xwidth):((i + 1) * xwidth),
                     (j * ywidth):((j + 1) * ywidth)] = maskval
                masks.append(mask)
        return np.array(masks)

    def gray2rgb(self, im):
        return np.concatenate(3 * (im[..., np.newaxis],), axis=-1)

    def explain_prediction_heatmap(self, im_flat, title='Heatmap', nmasks=(9, 7, 5, 3, 2)):
        im = skimage.color.gray2rgb(im_flat)
        im = np.array([im])
        assert im.shape == (1, 28, 28, 3)

        ci = PIL.Image.fromarray(np.uint8(im_flat * 255))  # instead of cropping, convert to PIL image format
        masks = np.concatenate([self.make_masks(im, n=i) for i in nmasks])
        masknorm = masks.sum(axis=0)

        preds = self.model.predict(im_flat)
        predidx = np.argsort(preds)[::-1]
        topclasses = {}
        for i, p in enumerate(predidx[:5]):
            print(p, preds[p])
            topclasses[i] = (p, p)

        # print first element of each tuple in topclasses - they are duplicates
        topclasses_reduced = {}
        for k in topclasses.keys():
            topclasses_reduced[k] = str(topclasses[k][0])
        print('Top classes: ', topclasses_reduced)

        heatmaps = np.zeros((5,) + im.shape[1:3])

        for m in tqdm.tqdm(masks):
            prediction = self.model.predict(im * self.gray2rgb(m))
            for c in range(5):
                clsnum, clsname = topclasses[c]
                heatmaps[c] += (prediction[clsnum] * m)
        for h in heatmaps:
            h = h / masknorm
        fig, axes = plt.subplots(2, 6, figsize=(10, 5))
        axes[0, 0].imshow(ci, cmap="gray"), axes[1, 0].imshow(ci, cmap="gray")  # convert to grayscale
        axes[0, 0].set_title(title)
        hide_axes(axes[0, 0]), hide_axes(axes[1, 0])
        predictions = np.sum(heatmaps, axis=(1, 2,))
        predictions /= predictions.max()
        for n, i in enumerate(np.argsort(predictions)[::-1][:5]):
            h = ((255 * heatmaps[i]) / heatmaps[i].max()).astype('uint16')
            h = skimage.exposure.equalize_adapthist(h)
            axes[0, n + 1].imshow(self.gray2rgb(h))
            maskim = np.squeeze(im[:, :, :, ::-1]) * \
                self.gray2rgb(h) * (0.5 + 0.5 * predictions[i])
            maskim -= maskim.min()
            maskim /= maskim.max()
            axes[1, n + 1].imshow(maskim)
            hide_axes(axes[0, n + 1]), hide_axes(axes[1, n + 1])
            axes[0, n + 1].set_title(str(topclasses[i][1]) +
                                     ': %0.1f%%' % (100 * predictions[i] / predictions.sum()))
        fig.tight_layout()
        # plt.show()
        # return heatmaps
        return fig  # return the figure obj instead of the heatmaps array, for saving
