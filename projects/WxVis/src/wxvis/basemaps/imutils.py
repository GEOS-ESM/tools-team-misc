import PIL
from PIL import Image, ImageChops, ImageEnhance, ImageDraw, ImageFont
from PIL.ImageColor import getcolor, getrgb
from PIL.ImageOps import grayscale, expand

def immask(src, value):

        data    = src.getdata()
        mask    = self.mask.getdata()
        img     = src.copy()
        newData = []

        for i in range(0,len(data)):
            idata = data[i]
            imask = mask[i]

            if imask[0] == value:
                newData.append(idata)
            else:
                newData.append((0,0,0,0))

        img.putdata(newData)

        return img

#------------------------------------------------------------------------------

def imfill(src, value, color):

        if self.is_clear(color): return src

        data    = src.getdata()
        mask    = self.mask.getdata()
        img     = src.copy()
        newData = []
        color   = tuple(color)

        for i in range(0,len(data)):
            idata = data[i]
            imask = mask[i]

            if imask[0] == value:
                newData.append(color)
            else:
                newData.append(idata)

        img.putdata(newData)

        return img

#------------------------------------------------------------------------------

def imtint(src, tint_color):

        if self.is_clear(tint_color): return src

        tint_color  = '#%02x%02x%02x'%tuple(tint_color)
        return self.image_tint(src, tint_color)

#------------------------------------------------------------------------------

def imbright(src, brightness):

        if brightness == 1.0: return src

        enhancer = ImageEnhance.Brightness(src)
        return enhancer.enhance(brightness)

#------------------------------------------------------------------------------

def imsat(src, saturation):

        if saturation == 1.0: return src

        enhancer = ImageEnhance.Color(src)
        return enhancer.enhance(saturation)

#------------------------------------------------------------------------------

def imcontrast(src, contrast):

        if contrast == 1.0: return src

        enhancer = ImageEnhance.Contrast(src)
        return enhancer.enhance(contrast)

#------------------------------------------------------------------------------

def is_solid(color): 
        if len(color) < 4: return True
        if color[3] > 0.0: return True
        return False

#------------------------------------------------------------------------------

def is_clear(color):
        return not self.is_solid(color)

#------------------------------------------------------------------------------

def image_tint(src, tint='#ffffff'):

        if isinstance(src,six.string_types):
            src = Image.open(src)
        if src.mode not in ['RGB', 'RGBA']:
            raise TypeError('Unsupported source image mode: {}'.format(src.mode))
        src.load()

        tr, tg, tb = getrgb(tint)
        tl = getcolor(tint, "L")  # tint color's overall luminosity
        if not tl: tl = 1  # avoid division by zero
        tl = float(tl)  # compute luminosity preserving tint factors
        sr, sg, sb = map(lambda tv: tv/tl, (tr, tg, tb))  # per component
                                                      # adjustments
        # create look-up tables to map luminosity to adjusted tint
        # (using floating-point math only to compute table)
        luts = (tuple(map(lambda lr: int(lr*sr + 0.5), range(256))) +
                tuple(map(lambda lg: int(lg*sg + 0.5), range(256))) +
                tuple(map(lambda lb: int(lb*sb + 0.5), range(256))))
        l = grayscale(src)  # 8-bit luminosity version of whole image
        if sys.version_info.major==2: mode_len=Image.getmodebands(src.mode)
        else: mode_len=len(src.getbands())
        if mode_len < 4:
            merge_args = (src.mode, (l, l, l))  # for RGB verion of grayscale
        else:  # include copy of src image's alpha layer
            a = Image.new("L", src.size)
            a.putdata(src.getdata(3))
            merge_args = (src.mode, (l, l, l, a))  # for RGBA verion of grayscale
            luts += tuple(range(256))  # for 1:1 mapping of copied alpha values

        return Image.merge(*merge_args).point(luts)

#------------------------------------------------------------------------------

def image_trim(im): 

        bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            return im.crop(bbox)
        return im

