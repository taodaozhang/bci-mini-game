"""Self-contained x64 Windows MVP, including native drag/drop library."""
from pathlib import Path
import os
import shutil
import sys
from PIL import Image,ImageDraw
import PyInstaller.__main__
from prepare_bundle import main as prepare

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]


def main():
    prepare()
    tcl=Path(sys.base_prefix)/'tcl';os.environ['TCL_LIBRARY']=str(tcl/'tcl8.6');os.environ['TK_LIBRARY']=str(tcl/'tk8.6')
    icon=HERE/'assets/app.ico'
    img=Image.new('RGBA',(256,256),'#102b3e');draw=ImageDraw.Draw(img)
    draw.rounded_rectangle((8,8,248,248),radius=38,fill='#102b3e')
    for x,y in [(52,156),(52,108),(100,108),(148,108)]:draw.rectangle((x,y,x+42,y+42),fill='#4ee1bd')
    draw.rectangle((176,47,218,79),fill='#ff595f');draw.rectangle((194,32,202,49),fill='#b97b4c');draw.rectangle((203,29,222,42),fill='#8dda69')
    img.save(icon,sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])
    dist=HERE.parent/'程序';dist.mkdir(exist_ok=True)
    PyInstaller.__main__.run([
        str(HERE/'snake_app.py'),'--onefile','--windowed','--noupx','--noconfirm','--name','脑电贪吃蛇_MVP_1.0',
        '--icon',str(icon),'--distpath',str(dist),'--workpath',str(PROJECT/'work/snake_exe_build'),'--specpath',str(HERE),
        '--paths',str(HERE),'--add-data',f'{HERE/"assets"}{os.pathsep}assets',
        '--collect-data','tkinterdnd2','--collect-binaries','tkinterdnd2',
        '--hidden-import','sklearn.svm._classes','--hidden-import','sklearn.discriminant_analysis',
        '--hidden-import','sklearn.neural_network._multilayer_perceptron','--hidden-import','sklearn.ensemble._forest',
        '--hidden-import','sklearn.pipeline','--hidden-import','sklearn.preprocessing._data','--hidden-import','sklearn.tree._classes',
        '--exclude-module','matplotlib','--exclude-module','pyedflib','--exclude-module','pandas','--exclude-module','IPython','--exclude-module','pytest'
    ])
    examples=dist/'示例数据';examples.mkdir(exist_ok=True)
    for name in ('A03E.mat','B01E.mat'):shutil.copy2(HERE.parent/'示例数据'/name,examples/name)
    shutil.copy2(HERE.parent/'使用说明.txt',dist/'使用说明.txt')
    print('MVP ready:',dist,flush=True)


if __name__=='__main__':main()
