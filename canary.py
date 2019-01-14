import thorpy, pygame, sys, requests, io
from PIL import Image, ImageMath, ImageOps
from math import erf, sqrt, cos, sin, pi

TAU = 2*pi

SIZE = (500, 500)
CAPTION = 'Canary Edge Detection Mockup'
INSERTER_SIZE = (300, 50)

scaledSize = 1200
scaledFilter = Image.ANTIALIAS
gaussStdev = 1.0
gaussSpread = 1.0
gaussKernalSize = 3
hysterisisLvl = 7
supLo = .5
supHi = 2.0
imgSize = 400



def loadImage(url):
    response = requests.get(url).content
    return Image.open(io.BytesIO(response))

def imgPySurf(img):
    return pygame.image.fromstring(img.tobytes('raw', 'RGB'), img.size, 'RGB')

def checkType(value, inst, loc, name, nParameter, extraValid=True, extraValidStr=''):
    def _checkType(_value, _inst, _loc, _name, _nParameter, _extraValid=True, _extraValidStr=''):
        pass
        #assert (isinstance(_value, _inst) if not (isinstance(_inst, list) or isinstance(_inst, tuple)) else any ((isinstance(_value, inst0) for inst0 in _inst))) and _extraValid, '{loc} {name} ({nParameter}{nParameterExt} parameter) must be an instance of {inst}{extraStr}: {value} (type={valueType})'.format(loc=_loc, name=_name, inst=_inst, value=_value, valueType=type(_value), nParameter=_nParameter+1, nParameterExt=('st', 'nd', 'rd', 'th')[min((_nParameter+9)%10, 3)], extraStr='' if len(_extraValidStr) == 0 else (' ({} caused failure)'.format(_extraValidStr)
    _checkType(loc, str, 'checkType', 'loc', 2)
    _checkType(name, str, 'checkType', 'name', 3)
    _checkType(nParameter, int, 'checkType', 'nParameter', 4)
    _checkType(extraValid, bool, 'checkType', 'extraValid', 5)
    _checkType(extraValidStr, str, 'checkType', 'extraValidStr', 6)
    _checkType(value, inst, loc, name, nParameter, extraValid, extraValidStr)

checkType(scaledSize, int, 'Globals', 'scaledSize', 0, scaledSize > 0, '>0')
checkType(scaledFilter, type(Image.ANTIALIAS), 'Globals', 'scaledFilter', 0, scaledFilter in (Image.ANTIALIAS, Image.NEAREST, Image.BICUBIC), 'must be Image.(ANTIALIAS, NEAREST, BICUBIC)')
checkType(gaussStdev, float, 'Globals', 'gaussStdev', 0, gaussStdev > 0.0, '>0')
checkType(gaussSpread, float, 'Globals', 'gaussSpread', 0, gaussSpread > 0.0, '>0')
checkType(gaussKernalSize, int, 'Globals', 'gaussKernalSize', 0, gaussKernalSize > 0, '>0')
checkType(hysterisisLvl, int, 'Globals', 'hysterisisLvl', 0, hysterisisLvl > 0, '>0')
checkType(supLo, float, 'Globals', 'supLo', 0, 0.0 <= supLo <= supHi, 'between 0 and supHi')
checkType(supHi, float, 'Globals', 'supHi', 0, supHi >= supLo, '>supLo')

def prepImg(img):
    checkType(img, Image.Image, 'prepImg', 'img', 0)
    return ImageMath.eval('float(a)/255.0', a=img)

def unprepImg(img):
    checkType(img, Image.Image, 'unprepImg', 'img', 0)
    return ImageMath.eval('convert(255.0*a, "L")', a=img)

def shrink(img):
    checkType(img, Image.Image, 'shrink', 'img', 0)
    w, h = img.size
    if scaledSize >= max(w, h):
        return img.copy()
    elif w >= h:
        return img.resize((scaledSize, scaledSize*h/w), scaledFilter)
    else:
        return img.resize((scaledSize*w/h, scaledSize), scaledFilter)

def manualConvolve(img, kernal, kernalSize):
    checkType(img, Image.Image, 'manualConvolve', 'img', 0)
    checkType(kernal, (list, tuple), 'manualConvolve', 'kernal', 1)
    checkType(kernalSize, int, 'manualConvolve', 'kernalSize', 2, kernalSize**2 == len(kernal), 'len == kernalSize**2')

    k = kernalSize//2
    fimg = ImageOps.expand(Image.new('F', img.size), k)
    brdImg = ImageOps.expand(img, k)
    for y in range(kernalSize):
        for x in range(kernalSize):
            fimg = ImageMath.eval('a+c*b', a=fimg, b=brdImg.offset(x-k, y-k), c=kernal[x+y*kernalSize])
    return ImageOps.crop(fimg, k)

def convolve(img, kernal, kernalSize=None):
    checkType(img, Image.Image, 'convolve', 'img', 0)
    checkType(kernal, (list, tuple), 'convolve', 'kernal', 1, len(kernal) > 0, 'len > 0')
    kernalSize = kernalSize if kernalSize is not None else int(sqrt(float(len(kernal))))
    checkType(kernalSize, int, 'convolve', 'kernalSize', 2, kernalSize**2 == len(kernal), 'len == kernalSize**2')
    
    return manualConvolve(img, kernal, kernalSize)

def gaussKernal():
    kernal = list()
    i, lim, 0, (2*gaussKernalSize+1)**1

    def G(x, y):
        x, y, spread = float(x), float(y), guassSpread/2.0
        spreads = tuple(((i % 2, i//2) for i in range(4)))
        spreads = tuple((tuple(float(2*elem-1) for elem in elems) for elems in spreads))
        return sum((sx*sy*erf((x+sx*spread)/gaussStdev)*erf((y+sy*spread)/gaussStdev) for sx, sy in spreads))/4.0

    for y in range(2*gaussKernalSize+1):
        for x in range(2*gaussKernalSize+1):
            kernal.append(G(x, y))
    return kernal

def gaussianBlur(img):
    checkType(img, Image.Image, 'gaussianBlur', 'img', 0)
    return convolve(img, gaussianKernal(), 2*gaussKernalSize+1)

def prewittOperator():
    prewittX, prewittY = tuple(), tuple()
    for y in range(3):
        for x in range(3):
            prewittX += (float(x-1)*(1+abs(y-1)),)
            prewittY += (float(y-1)*(1+abs(x-1)),)
    return prewittX, prewittY, 3

def getGradientVec(img, opX, opY, opSize):
    checkType(img, Image.Image, 'getGradientVec', 'img', 0)
    checkType(opX, (list, tuple), 'getGradientVec', 'opX', 1)
    checkType(opY, (list, tuple), 'getGradientVec', 'opY', 2)
    checkType(opSize, int, 'getGradientVec', 'opSize', 3, opSize**2==len(opX) and opSize**2==len(opY), 'opSize**2==len(opX) (={}) and opSize**2==len(opY) (={})'.format(len(opX), len(opY)))
    gx = convolve(img, opX, opSize)
    gy = convolve(img, opY, opSize)
    return gx, gy

def getGradientPolar(img, gx, gy, nSlices, angOff=0.0):
    checkType(gx, Image.Image, 'getGradientPolar', 'gx', 0)
    checkType(gy, Image.Image, 'getGradientPolar', 'gy', 1)
    checkType(nSlices, int, 'getGradientPolar', 'nSlices', 2, nSlices > 0, '>0')
    checkType(angOff, float, 'getGradientPolar', 'angOff', 3)

    norm = ImageMath.eval('(a*a+b*b)**0.5', a=gx, b=gy)
    angxs, angys = tuple(), tuple()
    for i in range(nSlices):
        angx, angy = cos(angOff + (2*i + 1)*tau/(2.0*nSlices)), sin(angOff + (2*i + 1)*tau/(2.0*nSlices))
        angxs += (ImageMath.eval('gx*ax-gy*ay', gx=gx, gy=gy, ax=angx, ay=angy),)
        angys += (ImageMath.eval('gy*ax-gx*ay', gx=gx, gy=gy, ax=angx, ay=angy),)
    return norm, angxs, angys

def nonMaximumSuppression(norms, angxs, angys):
    def genEvalStr(opStr, varList, large):
        format1 = lambda fstr: fstr.format(gt='min(1.0,{large}*max({},0.0))',
                gte='(1-min(1.0,{large}*(-min({},0.0))))',
                lt='min(1.0,{large}*(-min({},0.0)))',
                lte='(1-min(1.0,{large}*max({},0.0)))',
                ne='min(1.0,{large}*abs({}))',
                eq='(1-min(1.0,{large}*abs({})))',
                andc=')*(',
                orc=')+(',
                notc='1-')
        format2 = lambda fstr:'max(1.0,abs(({})))'.format(fstr.format(*varList, large=large))
        tmpStr = format1(opStr)
        return format2(tmpStr)
    
    masks = tuple()
    for i in range(8):
        evalOps = '{gt}{andc}{gte}{andc}{gt}'
        evalVars = ('x', 'y', '(x-y)')
        evalStr = '''
        (L*(x+abs(x))+
        2-abs(L*(x+abs(x))-2))*
        (1-(L*(abs(y)-y)+2-
        abs(L*(abs(y)-y)-2))/4)*
        (1-(L*(abs(x-y)-x+y)+2-
        abs(L*(abs(x-y)-x+y)-2))/4)/4
        '''.replace('\n','').replace('L', '765.0').replace('\t','').replace(' ','')

        masks += (ImageMath.eval(evalStr, x=angxs[i], y=angys[i]),)
    fmasks = tuple()
    for i in range(4):
        fmasks += (ImageOps.expand(ImageMath.eval('(a+b+abs(a-b))/2', a=masks[i], b=masks[i+4]),1),)
    print('masks commplete')

    shifts = ((0,0,1), (1,1,1), (2,1,0), (3,1,-1))
    bnorms = ImageOps.expand(norms, 1)
    nonmax = Image.new('F', bnorms.size)

    for i, shiftX, shiftY in shifts:
        shift0 = ImageOps.expand(norms, 1).offset(-shiftX, -shiftY)
        shift2 = ImageOps.expand(norms, 1).offset(shiftX, shiftY)
        evalStr = '''
        m*(1-(1-(L
        *abs(a-b)+
        1-abs(L*ab
        s(a-b)-1))
        /2)*(1-(L*
        abs(a-c)+1
        -abs(L*abs
        (a-c)-1))/
        2))*(1-(1-
        (1-(L*(a-b
        +abs(a-b))
        +2-abs(L*(
        a-b+abs(a-
        b))-2))/4)
        *(1-(L*(c-
        b+abs(c-b)
        )+2-abs(L*
        (c-b+abs(c
        -b))-2))/4
        ))*(1-(1-(
        L*(abs(a-b
        )-a+b)+2-a
        bs(L*(abs(
        a-b)-a+b)-
        2))/4)*(1-
        (L*(abs(c-
        b)-c+b)+2-
        abs(L*(abs
        (c-b)-c+b)
        -2))/4)))
        '''.replace('\n','').replace('L','765.0').replace(' ','').replace('\t','')
        fangMask = ImageMath.eval(evalStr, a=shift0, b=norms, c=shift2, m=fmasks[i])
        nonmax = ImageMath.eval('(a+b+abs(a-b))/2', a=nonmax, b=fangMask)
    return ImageOps.crop(nonmax, 1)

def hysteresis(img, minConnecting):
    ys = frozenset(range(img.size[1]))
    fimg = Image.new('F', img.size)
    checked = set()
    pix, fpix = img.load(), fimg.load()
    num = lambda x, y:x+y*img.size[0]
    isValid = lambda x, y:img.size[0] > x >= 0 and img.size[1] > y >= 0

    class pnt():
        def __init__(self, x, y):
            self.x, self.y = x, y
        def __hash__(self):
            return id(self)
    
    for x in range(img.size[0]):
        for y in ys:
            if num(x, y) not in checked:
                path, valid = set(), set()
                path.add(pnt(x, y))
                while len(path) > 0:
                    pos = path.pop()
                    px, py = pos.x, pos.y
                    if isValid(px, py):
                        posNum = num(px, py)
                        if posNum not in checked:
                            checked.add(posNum)
                            if pix[px, py] > 0:
                                valid.add(pnt(px, py))
                                for nx in range(-1, 2):
                                    for ny in range(-1, 2):
                                        path.add(pnt(px + nx, py + ny))
                if len(valid) >= minConnecting:
                    for pos in valid:
                        fpix[pos.x, pos.y] = 1.0
    return fimg

def doubleThreshhold(nonmax, valid, low, high):
    return ImageMath.eval('v*x*(1-(L*(abs(x-l)-x+l)+2-abs(L*(abs(x-l)-x+l)-2))/4)*(1-(L*(abs(h-x)-h+x)+2-abs(L*(abs(h-x)-h+x)-2))/4)'.replace('L', '765.0').replace('l', str(low)).replace('h', str(high)), v=valid, x=nonmax)


def processImg():
    print('processing image')


def exitApp():
    pygame.display.quit()
    pygame.quit()
    sys.exit()

class ImageTransPainter(thorpy.painters.painter.Painter):
    def __init__(self, _img, size=None, clip=None, pressed=False, hovered=False,):
        super(ImageTransPainter, self).__init__((_img.size[0] + 2, _img.size[1] + 2), clip, pressed, hovered)
        self.img = _img
        self.w, self.h = _img.size[0], _img.size[1]

    def get_surface(self):
        surface = pygame.Surface(self.size, flags=pygame.SRCALPHA).convert_alpha()
        _img = imgPySurf(self.img)
        pygame.draw.rect(surface, (0,)*3, (0, 0, self.w + 2, self.h + 2), 1)
        surface.blit(_img, (1, 1))
        surface.set_clip(self.clip)
        return surface
    
    def swapImg(self, img):
        self.set_size((img.size[0] + 2, img.size[1] + 2))
        self.img = img
        self.w, self.h = img.size[0], img.size[1]
        self.refresh_clip()

def getImageTrans(img):
    container = thorpy.Element()
    container.set_painter(ImageTransPainter(img))
    container.finish()
    return container

def swapImageTrans(container, img):
    container.change_painter(ImageTransPainter(img))
    #thorpy.store(background, [inserter, runBtn, imgBox1, imgBox2])
    container.total_unblit()
    container.blit()
    container.total_update()
    return container

def adjustImgSize(img, targetSize):
    if img.size[0] > img.size[1]:
        return img.resize((int(targetSize), int(targetSize*img.size[1]/img.size[0])))
    else:
        return img.resize((int(targetSize*img.size[0]/img.size[1]), int(targetSize)))

def showImg(img, targetSize, label, desc):
    img = adjustImgSize(img, targetSize)
    labelObj = thorpy.OneLineText.make(label)
    imgBox = getImageTrans(img)
    descObj = thorpy.OneLineText.make(desc)
    box = thorpy.make_ok_box([labelObj, imgBox, descObj])
    
    launcher = thorpy.Launcher(box)
    react = thorpy.ConstantReaction(thorpy.constants.THORPY_EVENT,
            launcher.unlaunch,
            {'id':thorpy.constants.EVENT_DONE, 'el':box},
            {'what':thorpy.constants.LAUNCH_DONE})
    box.add_reaction(react)
    launcher.launch()

def showImgBtn(targetSize, label, desc):
    global imgTable
    imgTable[label] = imgNone
    
    def reactFn(elem):
        x, y = pygame.mouse.get_pos()
        rect = elem.get_rect()
        if rect.left < x < rect.right and rect.top < y < rect.bottom:
            print('reaction!')
            showImg(imgTable[label], targetSize, label, desc)

    launchBtn = thorpy.Pressable.make(text=label)
    react = thorpy.ConstantReaction(reacts_to=pygame.MOUSEBUTTONDOWN, reac_func=reactFn, params={'elem':launchBtn})
    launchBtn.add_reaction(react)

    return launchBtn

def keyEvent(event):
    global isA, isB, isC, runBtn
    if event.key in (pygame.K_KP_ENTER, pygame.K_RETURN):
        processImg()
    elif event.key == pygame.K_ESCAPE:
        exitApp()
    elif event.key == pygame.K_n:
        img = loadImage('http://www.thorpy.org/images/thorpy.png')
        if isC:
            img = adjustImgSize(imgNone, 300)
        isC = not isC
        showImg(img, 300, 'Test Image', 'Test Image description')

imgTable = dict()

reaction = thorpy.Reaction(reacts_to=pygame.KEYDOWN, reac_func=keyEvent)

app = thorpy.Application(size=SIZE, caption=CAPTION)

inserter = thorpy.Inserter.make(name='Image URL :', value='URL HERE', size=INSERTER_SIZE)

runBtn = thorpy.make_button('Run')

imgNone = Image.open('noImg.png')

imgBox1 = getImageTrans(adjustImgSize(imgNone, 300))
imgBox2 = getImageTrans(adjustImgSize(imgNone, 300))

showOriginal = showImgBtn(imgSize, 'Original', 'The original image.')
showGray = showImgBtn(imgSize, 'Grayscale', 'Black and white version.')
showGauss = showImgBtn(imgSize, 'Gaussian Blur', 'Guassian blur filter applied.')
showNorms = showImgBtn(imgSize, 'Gradient Norms', 'Norm of gradients applied.')
showNonMax1 = showImgBtn(imgSize, 'Non-maximum Suppression', 'Desc. here.')
showNonMax2 = showImgBtn(imgSize, 'Non-maximum Suppression over Norms', 'Desc. here.')
showThresh = showImgBtn(imgSize, 'Double Threshold', 'Desc. here.')
showHysteresis = showImgBtn(imgSize, 'Hysteresis [FINAL VERSION]', 'Desc. here.')

def processImage(img):
    pass

background = thorpy.Background.make((255,)*3, elements=[inserter, runBtn, showOriginal, showGray, showGauss, showNorms, showNonMax1, showNonMax2, showThresh, showHysteresis])
thorpy.store(background, [inserter, runBtn, showOriginal, showGray, showGauss, showNorms, showNonMax1, showNonMax2, showThresh, showHysteresis])

background.add_reaction(reaction)

thorpy.Menu(background).play()
app.quit()
