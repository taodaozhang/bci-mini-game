"""Build a self-contained Windows exe using the current scientific environment."""
from pathlib import Path
import os
import sys
import shutil
from PIL import Image, ImageDraw
import PyInstaller.__main__
from prepare_bundle import main as prepare

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]


def main():
    prepare()
    # Explicit Tcl paths also let PyInstaller inspect the bundled Python runtime.
    tcl=Path(sys.base_prefix)/'tcl'
    os.environ['TCL_LIBRARY']=str(tcl/'tcl8.6');os.environ['TK_LIBRARY']=str(tcl/'tk8.6')
    icon=HERE/'assets/app.ico'
    img=Image.new('RGBA',(256,256),(16,42,67,255));draw=ImageDraw.Draw(img)
    draw.rounded_rectangle((12,12,244,244),radius=54,fill='#102a43')
    draw.line([(36,128),(70,128),(88,80),(110,179),(135,103),(153,128),(220,128)],fill='#46d2cc',width=13,joint='curve')
    draw.polygon([(35,128),(61,110),(61,146)],fill='white');draw.polygon([(221,128),(195,110),(195,146)],fill='white')
    img.save(icon,sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])
    dist=HERE.parent/'程序';dist.mkdir(parents=True,exist_ok=True)
    work=PROJECT/'work/eeg_exe_build'
    PyInstaller.__main__.run([
        str(HERE/'eeg_app.py'),'--onefile','--windowed','--noupx','--noconfirm',
        '--name','脑电左右手识别器','--icon',str(icon),'--distpath',str(dist),
        '--workpath',str(work),'--specpath',str(HERE),'--paths',str(HERE),
        '--add-data',f'{HERE / "assets"}{os.pathsep}assets',
        '--hidden-import','sklearn.svm._classes','--hidden-import','sklearn.discriminant_analysis',
        '--hidden-import','sklearn.neural_network._multilayer_perceptron',
        '--hidden-import','sklearn.ensemble._forest','--hidden-import','sklearn.pipeline',
        '--hidden-import','sklearn.preprocessing._data','--hidden-import','sklearn.tree._classes',
        '--exclude-module','matplotlib','--exclude-module','pyedflib',
        '--exclude-module','pandas','--exclude-module','IPython','--exclude-module','pytest',
    ])
    examples=dist/'示例数据';examples.mkdir(exist_ok=True)
    for source in [PROJECT/'outputs/external_eeg_tests/data/raw/bnci001/A03E.mat',PROJECT/'outputs/eeg_project/data/raw/B01E.mat']:
        shutil.copy2(source,examples/source.name)
    shutil.copy2(HERE.parent/'使用说明.txt',dist/'使用说明.txt')
    print('Build complete:',dist,flush=True)


if __name__=='__main__':main()
