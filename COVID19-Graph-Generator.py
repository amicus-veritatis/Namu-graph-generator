import time
from bs4 import BeautifulSoup as Soup
from requests import get, post
import pandas as pd
import pandas_read_xml as pdx
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import datetime as dt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import itertools as itt

#############################
url = 'http://openapi.data.go.kr/openapi/service/rest/Covid19/getCovid19InfStateJson'
ServiceKey = #######################
##############################



matplotlib.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_rows', 500)
path = '/NotoSansCJKkr-Black.otf'
path2 = '/NotoSansCJKkr-Bold.otf'
path3 = '/NotoSansCJKkr-Medium.otf'
path4 = '/NotoSansCJKkr-Regular.otf'
fontprop = fm.FontProperties(fname=path)
fontprop3 = fm.FontProperties(fname=path2,size=84)
fontprop2 = fm.FontProperties(fname=path,size=48) #legend
fontprop4 = fm.FontProperties(fname=path2,size=48)
fontprop4 = fm.FontProperties(fname=path2,size=56)
fontprop5 = fm.FontProperties(fname=path3,size=40)
fontprop6 = fm.FontProperties(fname=path,size=48)
fontprop7 = fm.FontProperties(fname=path2,size=40)
font1 = fontprop.get_name()
font2 = fontprop2.get_name()

def GenPlot(DataIn):
    left = dt.date(2020, 1,20)
    right = dt.date.today()
    tick_spacing = 5000
    nDay = right if now.hour<10 else right+dt.timedelta(days=1)
    days = mdates.drange(left,nDay,dt.timedelta(days=1))
    fig, [ax,timeline] = plt.subplots(nrows=2, ncols=1, sharex='col', gridspec_kw={'height_ratios': [30, 12], 'wspace' : 0, 'hspace':0},figsize=(48,36),constrained_layout=True, facecolor='#cad9e1')
    params = {"figure.facecolor": "#cad9e1",
              "axes.facecolor": "#cad9e1",
              "axes.grid" : True,
              "axes.grid.axis" : "y",
              "grid.color"    : "#ffffff",
              "grid.linewidth": 2,
              "axes.spines.left" : False,
              "axes.spines.right" : False,
              "axes.spines.top" : False,
              "ytick.major.size": 0,     
              "ytick.minor.size": 0,
              "xtick.direction" : "in",
              "xtick.major.size" : 8,
              "xtick.color"      : "#191919",
              "axes.edgecolor"    :"#191919",
              "text.color" : "#4a4a4a",
              'axes.labelcolor': "#4a4a4a",
              "axes.prop_cycle" : plt.cycler('color',
                                    ['#006767', '#ff7f0e', '#2ca02c', '#d62728',
                                     '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                                     '#bcbd22', '#17becf'])}
    plt.rcParams.update(params)
    ax.set_title('대한민국 내 COVID-19 현황', fontproperties=fontprop3, pad=12, fontdict=dict(color="#4a4a4a"))
    ax2 = ax.twinx()
    line1 = ax.plot(days, DataIn['decideCnt'].astype(int).tolist(), color="#244747")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y\n%m')) 
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))
    ax.set_facecolor('#cad9e1')
    ax.tick_params(axis="y", labelsize=48)
    ax2.tick_params(axis="y", labelsize=48)
        ######trans#######
    dd = ['2020-02-23','2020-05-05','2020-08-16', '2020-09-13','2020-11-24']
    dd = [dt.datetime.strptime(d, "%Y-%m-%d") for d in dd]
    dd = [mdates.date2num(d) for d in dd]
    dd.append(mdates.date2num(right))
    for (i,j) in zip(dd[0::2], dd[1::2]):
      ax.axvspan(i,j,facecolor='#3E647D', alpha=0.25)
    ax.fill_between(days, DataIn['decideCnt'].astype(int).tolist(), color="#336666",
    alpha=0.5,label="COVID-19 누적 확진자")
    ax.fill_between(days, DataIn['clearCnt'].astype(int), color="#8abbd0",
                 label="COVID-19 누적 격리해제자")
    ax.fill_between(days,np.pad([],(0,len(days)), 'constant', constant_values=0),color="#823c5a",
                 alpha=1,label="COVID-19 일일 확진자")
    
    #######
    ax.annotate('Test', xy=(days[29], DataIn['decideCnt'].astype(int).iloc[29]), xytext=(days[29], DataIn['Event'].astype(int).iloc[29]), 
            textcoords='data', arrowprops=dict(arrowstyle='-|>'))
    ax.grid(True,'major', 'y',color='#ffffff', linestyle='-', linewidth=2, alpha=1)
    ax2.grid(True,'major', 'y',color='#ffffff', linestyle='-', linewidth=2, alpha=0.5)
    for axs in [ax,timeline]:
        axs.set_frame_on(False)
    ax2.stem(days, DataIn['dConf'].tolist(), "#823c5a",markerfmt=" ", basefmt="#ffffff", use_line_collection=True,label="COVID-19 일일 확진자")
    ax.set_aspect('auto')
    ax.legend(bbox_to_anchor=(0.5, -0.02), loc='upper center', ncol=3,fancybox=True, prop=fontprop2,frameon=False)

    ax.xaxis.set_label_position('top')
    string = "(" + dt.datetime.now().strftime("%Y/%m/%d")+ " 기준)"
    
    ax.set_xlabel(string, fontproperties=fontprop4, labelpad =28)

    ##### Timeline #########
    txts = ["신천지 집단감염",'감염병 위기경보\n\'심각\' 격상','강화된\n사회적 거리 두기 실시','사회적 거리 두기 종료\n생활 속 거리 두기 전환', '수도권 교회 집단감염','사회적 거리 두기\n2단계 격상','사회적 거리 두기\n2.5단계 격상','사회적\n거리 두기\n완화\n(2.5→2)','사회적 거리 두기\n완화\n(2→1)',
            '3차 대유행','사회적 거리 두기\n2단계 격상','']
    txts = list(reversed(txts))
    print(txts)
    dates = ['2020-11-24','2020-11-20', '2020-10-12','2020-09-13','2020-08-30','2020-08-16','2020-08-15', '2020-05-05','2020-03-22','2020-02-23', '2020-02-18']
    dates = [dt.datetime.strptime(d, "%Y-%m-%d") for d in dates]
    dates.append(left)
    dates.insert(0,right)
    print(dates)
    levels=[0,1,-0.5,-1,-1,1,-0.25,-1,-0.5,-1,1,-1,0]
    hets=[]
    levels = np.flip(levels).tolist()
    print(levels)
# 9일부터 주석 풀기
    timeline.tick_params(axis="x", labelsize=40)
    timeline.xaxis.set_major_formatter(mdates.DateFormatter('%Y\n%m')) 
    timeline.vlines(dates, 0, levels, color="#e3120b")
    timeline.plot(dates, np.zeros_like(dates), "-o",
        color="k", markerfacecolor="w", markersize=20)  # Baseline and markers on it.
#annotate line
    ha = ["right","left","center"]
    ha = ["center","left","left","center","right","right","left","center","right"]
    he = itt.cycle(ha)
    for d, l, r in zip(dates, levels, txts):
      timeline.annotate(r, xy=(d, l), xytext=(-3, np.sign(l)*3), textcoords="offset points", verticalalignment="bottom" if l > 0 else "top", ha="center"if l > 0 else next(he), fontproperties=fontprop6 if l>0 else fontprop7)
    timeline.text(0, -0.24, "자료 : 질병관리청", transform=timeline.transAxes,
        ha="left", va="bottom", color="#4a4a4a",
        fontproperties=fontprop4)
    timeline.set_ylim([-2,2])
    timeline.get_yaxis().set_visible(False)
    timeline.set_facecolor("#cad9e1")
    timeline.grid(True,'minor', 'x', color="#ffffff", alpha=1)
    fig2 = plt.gcf()
    fig2.savefig('test2png.png',dpi=100)
    fig2.show()



class req:
    
    def __init__(self,PageNo=0, Rows=500,DateFrom=20200310,DateTo=0):
        DateTo = time.strftime('%Y%m%d',time.localtime(time.time()))
        query = {'serviceKey': ServiceKey, 'PageNo':PageNo,'numOfRows':Rows,'startCreateDt':DateFrom,'endCreateDt':DateTo}
        self.resp = get(url=url,params=query)
        


a = req(PageNo=1, DateFrom=20200311)
response = a.resp.text
df0 = pd.read_csv("https://pastebin.com/raw/G1tdJpF6")
df = pdx.read_xml(response,['response','body','items','item'])
df = df.drop_duplicates('stateDt', keep='first')
now = dt.datetime.now()
sp = np.datetime_as_string(np.arange('2020-03-11',(now).strftime("%Y-%m-%d"),dtype='datetime64[D]'),unit='D')
sp = np.flip(sp) ###############
df['stateDt'] = sp
df = df.iloc[::-1]
df = pd.concat([df0,df],ignore_index=True)
df = df.drop_duplicates('stateDt', keep='first')
df['dConf'] = df['decideCnt'].astype(int).diff()
df['dConf'].iloc[0]=0
df['dConf'].iloc[51]=333
df['dConf'] = np.abs(df['dConf'].astype(int).tolist())
df['Event'] = np.pad(np.array([]),(0,len(df['stateDt'].tolist())), 'constant', constant_values = 0)
df['Eventline'] = np.empty(len(df['stateDt'].tolist()), dtype="<U10")
df['Event'].iloc[29]=1
df['Eventline'].iloc[29]='신천지 집단감염'
df['Event'].iloc[206]=-1
df['Eventline'].iloc[206]='수도권 교회 집단감염'
df['Event'].iloc[305]=-1
df['Eventline'].iloc[305]='3차 대유행'
for x,y,z in zip(range(46,51),[518,483,367,248,131],[6284,6767,7134,7382,7513]):
    df['dConf'].iloc[x]=y
    df['decideCnt'].iloc[x]=z
for x,y,z in zip(range(323,326),[592,671,680],[38746,39417,40097]):
    df['dConf'].iloc[x]=y
    df['decideCnt'].iloc[x]=z
# https://plotly.com/python/text-and-annotations/#text-font-as-an-array--styling-each-text-element
key_colums = ['stateDt','decideCnt','clearCnt','dConf', 'Event', 'Eventline']
data = Soup(response,'lxml-xml')
Datetime = reversed(data.find_all('stateDt'))
Confirm = reversed(data.find_all('decideCnt'))
Release = reversed(data.find_all('clearCnt'))
Death_toll = reversed(data.find_all('deathCnt'))
CARE = reversed(data.find_all('careCnt'))
Seq = reversed(data.find_all('seq'))
oConfirm = [i for i in Confirm]
#dConfirm = np.append(np.array([1]),np.array(oConfirm))
DateIn = sp
GenPlot(df[key_colums].copy())
#for date, conf, rlse, dead, care, seq in zip(Datetime,Confirm,Release,Death_toll,CARE,Seq):
#    print('{0}\t\t{1}\t{2}\t{3}'.format(str(int(date.string)-1),conf.string,dead.string,rlse.string))

