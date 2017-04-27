# Classifier demo
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import joblib
import urllib
import json
import io
import PIL
import tqdm
import skimage.exposure
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

datadir = os.path.expanduser(os.path.join('~', '.msia-deeplearning'))
if not os.path.exists(datadir):
    os.makedirs(datadir)
memory = joblib.Memory(os.path.join(datadir, 'joblib'), verbose=0)


def RetrieveURL(request):
    connection = urllib.request.urlopen(request)
    data = connection.read()
    connection.close()
    return data


@memory.cache
def RetrieveJSON(request): return json.loads(RetrieveURL(request).decode())


def GetAllImageIds(page=1, ids=[]):
    if page == 1:
        print('Loading visual genome data, please wait...')
    data = RetrieveJSON(
        'http://visualgenome.org/api/v0/images/all?page=' + str(page))
    if data['next'] is None:
        return ids
    return GetAllImageIds(page + 1) + data['results']


def GetImage(url):
    imgdata = io.BytesIO(RetrieveURL(url))
    return PIL.Image.open(imgdata)


def GetImageByID(imgid=None):
    imgid = imgid or random.choice(GetAllImageIds())
    url = RetrieveJSON(
        'http://visualgenome.org/api/v0/images/%d' % imgid)['url']
    print('Downloading image %d: %s' % (imgid, url))
    return GetImage(url)


def crop_img(img):
    if img.width > img.height:
        w = int(224 * img.width / img.height)
        img = img.resize((w, 224,))
        return img.crop((w / 4, 0, w / 4 + 224, 224,))
    else:
        h = int(224 * img.height / img.width)
        img = img.resize((224, h,))
        return img.crop((0, h / 4, 224, h / 4 + 224,))


def test(img=None):
    img = img or GetImageByID()
    img = crop_img(img)
    plt.imshow(img), plt.xticks([]), plt.yticks([]), plt.show()
    print('Classifying')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds, top=5)[0])


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

    def make_masks_modified(self, im, n=8, maskval=0.0):
        masks = []
        # original input has dimension (1, 224, 224, 3)
        # i.e. (1 image, x, y, RGB)
        xwidth, ywidth = int(np.ceil(im.shape[0] / n)), int(np.ceil(im.shape[1] / n))
        for i in range(n):
            for j in range(n):
                mask = np.ones(im.shape[0:2])
                mask[(i * xwidth):((i + 1) * xwidth),
                     (j * ywidth):((j + 1) * ywidth)] = maskval
                masks.append(mask)
        return np.array(masks)

    def gray2rgb(self, im): return np.concatenate(
        3 * (im[..., np.newaxis],), axis=-1)

    def explain_prediction_heatmap(self, im_flat, title='Heatmap', nmasks=(9, 7, 5, 3, 2)):
        im = skimage.color.gray2rgb(im_flat)
        im = np.array([im])
        assert im.shape == (1, 28, 28, 3)
        
        # instead of cropping, convert to PIL image format
        ci = PIL.Image.fromarray(np.uint8(im_flat * 255))
        
        masks = np.concatenate([self.make_masks(im, n=i) for i in nmasks])
        masknorm = masks.sum(axis=0)

        preds = model.predict(im_flat)
        # predicted_num = str(preds.argmax() - 1)
        
        predidx = np.argsort(preds)[::-1]
        topclasses = {}
        for i, p in enumerate(predidx[:5]):
            print(p, preds[p])
            topclasses[i] = (p, p)
        print('Top classes: ', topclasses)

        heatmaps = np.zeros((5,) + im.shape[1:3])

        for m in tqdm.tqdm(masks):
            prediction = self.model.predict(im * self.gray2rgb(m))
                  
            for c in range(5):
                clsnum, clsname = topclasses[c]
                heatmaps[c] += (prediction[clsnum] * m)
        for h in heatmaps:
            h = h / masknorm
        fig, axes = plt.subplots(2, 6, figsize=(10, 5))
        axes[0, 0].imshow(ci), axes[1, 0].imshow(ci)
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
        plt.show()
        return heatmaps


if __name__ == "__main__":
    
    model = Model(weights)
    heatmap = Heatmap(model)
    img = X[:, 0].reshape(28, 28)
    
    h = heatmap.explain_prediction_heatmap(img, nmasks=(4, 5))

#    #%% Higher resolution heatmap
#    h = heatmap.explain_prediction_heatmap(img, nmasks=(3, 4, 5, 7))
