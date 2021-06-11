import time
from bs4 import BeautifulSoup as Soup
from requests import get, post
import math
import pandas as pd
import json
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.offsetbox import AnchoredText
import datetime as dt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
from matplotlib.collections import PolyCollection
import itertools as itt
import pprint
#############################
url = 'https://api.odcloud.kr/api/15077756/v1/vaccine-stat'
ServiceKey = #########################################################################################
PopOfSK = 51829023
##############################

'''
To-do : 타임라인 구현
데이터 출처 : http://m.bosa.co.kr/news/articleView.html?idxno=2147974
https://www.javaer101.com/en/article/19770125.html
https://stackoverflow.com/questions/44518170/how-to-draw-a-bar-timeline-with-matplotlib
https://matplotlib.org/stable/gallery/lines_bars_and_markers/broken_barh.html
https://matplotlib.org/stable/gallery/lines_bars_and_markers/timeline.html#sphx-glr-gallery-lines-bars-and-markers-timeline-py
https://cm.asiae.co.kr/article/2021042610521646515
'''

matplotlib.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_rows', 500)
path = './NotoSansCJKkr-Black.otf' ######## 경로 직접 설정 필요
path2 = './NotoSansCJKkr-Bold.otf' ##########################
path3 = './NotoSansCJKkr-Medium.otf' ########################
path4 = './NotoSansCJKkr-Regular.otf' #######################
fontprop = fm.FontProperties(fname=path)
fontprop3 = fm.FontProperties(fname=path2,size=145)
fontprop2 = fm.FontProperties(fname=path,size=64) #legend
fontprop4 = fm.FontProperties(fname=path2,size=48)
fontprop4 = fm.FontProperties(fname=path2,size=72)
fontprop5 = fm.FontProperties(fname=path3,size=54)
fontprop6 = fm.FontProperties(fname=path,size=48)
fontprop7 = fm.FontProperties(fname=path,size=40)
fontprop8 = fm.FontProperties(fname=path,size=88)
font1 = fontprop.get_name()
font2 = fontprop2.get_name()
matplotlib.rcParams["font.family"] = font2
    
class mv_avg:
    def __init__(self, data=None, period=7, color=None):
        if data.empty:
            raise ValueError("Data가 비어있습니다.\n")
        self.color = color
        self.period = period
        self.data = data
    def sma(self):
        print("SMASMASMASMA\n")
        result = self.data.rolling(window=self.period).mean()
        print("SMA!" + str(result.tolist()))
        return result
    def ema(self):
        result = self.data.ewm(span=self.period, adjust=False).mean()
        return result

def PeriodEmphasizer(Axis=None, DateFrom=None, DateTo=None, Color="#3E647D", Alpha=0.25, isDateToCurrent=False):
    if not all(x is not None for x in (Axis, DateFrom, DateTo)):
        raise ValueError("Every values has to be filled\n")
    elif isDateToCurrent:
        DateTo = nTime.strftime("%Y-%m-%d")
    Date = [dt.datetime.strptime(d, "%Y-%m-%d") for d in zip(DateFrom, DateTo)]
    Date = [mdates.date2num(d) for d in Date]
    Axis.axvspan(Date[0],Date[1],facecolor=Color, alpha=Alpha)
    
def LineEmphasizer(Axis=None, Date=None, Data=None, Color=None, LineWidth=None, Label=None):
    if not all(x is not None for x in (Axis, Date, Color, LineWidth)):
        raise ValueError("Every values has to be filled\n")
    elif Data.size == None:
        raise ValueError("Data가 비어있습니다.\n")
    Axis.plot(Date, Data.astype(int).tolist(), color=Color, linewidth=LineWidth, label=Label)


def millions(x, pos):
    """The two args are the value and tick position."""
    return '{:1.1f}M'.format(x*1e-6)

def mans(x, pos):
    """The two args are the value and tick position."""
    if x ==0:
        return 0
    else:
        return '{:1.0f}만'.format(x*1e-4)

def PeriodLocator(ax, start, end):
    locs = [mdates.date2num(start)] + list(ax.get_xticks()) + [mdates.date2num(end)]
    locator = matplotlib.ticker.FixedLocator(locs)
    return locator

def TickAdjust(ax=None, tickAxis="Y", recent=None, Min=0):
    """
    425만 -> 500만 같이 Data의 lim을 변화시켜 Tick을 조정시키는 함수
    log10(a*10^M+b*10^(M-1)....) = M + log10(a+b*10^(-1)+....)
    """
    M = math.log10(recent)
    A = math.pow(10,M - math.floor(M))
    A = math.ceil(A)
    M = math.floor(M)
    Max = A * math.pow(10,M)
    lim_range = [Min, Max]
    if(tickAxis=="Y"):
        ax.set_ylim(lim_range)
    elif(tickAxis=="X"):
        ax.set_xlim(lim_range)
    else:
        raise(ValueError("tickAxis is neither Y nor X"))



def GenPlot(DataIn):
    left = dt.date(2021, 2,26)
    right = dt.date.today()
    tick_spacing = 5000
    nDay = right if now.hour<10 else right+dt.timedelta(days=1)
    days = mdates.drange(left,nDay,dt.timedelta(days=1))
    days = days[:-1] if len(days) != len(DataIn.iloc[::]) else days
    print("len(DataIn.iloc[::]) : " + str(len(DataIn.iloc[::])) + "\n")
    fig, [ax,timeline] = plt.subplots(nrows=2, ncols=1, sharex='col', gridspec_kw={'height_ratios': [36, 0.001], 'wspace' : 0, 'hspace':0},figsize=(48,32),constrained_layout=True, facecolor='#fafafa') ### [36,20]
    params = {"figure.facecolor": "#fafafa",
              "axes.facecolor": "#fafafa",
              "axes.grid" : True,
              "axes.grid.axis" : "y",
              "grid.color"    : "#ffffff",
              "grid.linewidth": 10,
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
              "font.family" : font1,
              'axes.labelcolor': "#4a4a4a",
              "axes.prop_cycle" : plt.cycler('color',
                                    ['#006767', '#ff7f0e', '#2ca02c', '#d62728',
                                     '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                                     '#bcbd22', '#17becf'])}
    plt.rcParams.update(params)
    ######RatAx : 비율#######
    '''
        1차 : totalFirstCnt / PopOfSK * 100 %
        2차 : totalSecondCnt / PopOfSk * 100 %
    '''
    ax.set_title('대한민국 내 COVID-19 백신 접종 현황', fontproperties=fontprop3, pad=12, fontdict=dict(color="#191919"))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y\n%m'))

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.yaxis.set_major_formatter(mans)
    TickAdjust(ax=ax, tickAxis="Y", recent=DataIn['totalSecondCnt'].iloc[-1]+DataIn['totalFirstCnt'].iloc[-1], Min=0)

    ###ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))
    ax.set_facecolor('#fafafa')
    ax.yaxis.tick_right()

    for label in ax.get_yticklabels():
        label.set_fontproperties(fontprop4)
        label.set_color('#4a4a4a')
        ######trans#######
    ax.grid(True,'major', 'y',color='#B6B6B6', linestyle='-', linewidth=3, alpha=1)
    #ffffff 3E647D
    ax.set_axisbelow(True)
    ax.fill_between(days, DataIn['totalFirstCnt'].astype(int).tolist(), color="#e3120b", label="1회차 접종자")
    ax.fill_between(days, (np.array(DataIn['totalSecondCnt'].astype(int).tolist())+np.array(DataIn['totalFirstCnt'].astype(int).tolist())).tolist(), color="#FBB4B1", label="2회차 접종자")
    ax.fill_between(days, DataIn['totalFirstCnt'].astype(int).tolist(), color="#e3120b")
    LineEmphasizer(ax,days,DataIn['totalFirstCnt'],'#fafafa',3)
    '''
    LineEmphasizer(ax,days,np.array(DataIn['totalSecondCnt'].astype(int).tolist())+np.array(DataIn['totalFirstCnt'].astype(int).tolist()),'#fafafa',3)
    '''
    #######

    for axs in [ax,timeline]:
        axs.set_frame_on(False)
    ax.set_aspect('auto')
    ax.xaxis.set_label_position('top')
    string = "(" + nTime.strftime("%Y/%m/%d")+ " 기준)\n\n"
    ax.set_xlabel(string, fontproperties=fontprop4, labelpad =28, color="#4a4a4a")
    timeline.set_xlim([left,right])
    timeline.tick_params(axis="x", labelsize=40)
    timeline.xaxis.set_major_locator(ticker.AutoLocator())
    timeline.xaxis.set_major_locator(PeriodLocator(timeline,left,nTime))
    timeline.tick_params(axis="x", which='major', length=30, width=8, labelsize=56)
    timeline.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=MO,interval=2))
    timeline.xaxis.set_major_formatter(mdates.DateFormatter('%Y\n%m/%d'))
    plt.setp(textbox.patch, facecolor='#fafafa', alpha=1)
    ax.add_artist(textbox)
    plt.annotate('자료 : 질병관리청', (0,0), (0,-180), fontproperties=fontprop4,
             xycoords='axes fraction', textcoords='offset points', va='top')
    timeline.set_ylim([-2,2])
    timeline.yaxis.tick_left()
    timeline.get_yaxis().set_visible(False) ############ 
    timeline.set_facecolor("#fafafa")
    right = (np.datetime64('today').astype('datetime64[M]') + np.timedelta64(5, 'W')).astype('datetime64[D]')
    mon = np.arange(np.datetime64('2021-02-21'), right, np.timedelta64(1, 'W'), dtype='datetime64[W]')
    mon = mon.tolist()
    for (i,j) in zip(mon[1::2], mon[2::2]):
        timeline.axvspan(i,j,facecolor='#FBB4B1', alpha=1)
 #   timeline.grid(True,'minor', 'x', color="#4a4a4a", alpha=0.5, linewidth=3)
    string = "대한민국 내 COVID-19 백신 접종 현황 ("
    Dtime = (nTime).strftime("%m.%d")
    string2 = string + Dtime + " 기준)"
    FDRate = DataIn['PopRatio-OneDose'].iloc[-1] * 100
    SDRate = DataIn['PopRatio-FullyVaccinated'].iloc[-1] * 100
    FDRate = "%0.1f%%" % FDRate
    SDRate = "%0.1f%%" % SDRate
    RatioString = "인구 대비 접종률: 1회차 %s, 2회차 %s" % (FDRate, SDRate)

    ###
    timeline.set_aspect('auto')
    timeline.xaxis.set_label_position('top')
    timeline.set_xlabel(RatioString, fontproperties=fontprop8, labelpad=96, color="#4a4a4a")

    ###

    ax.text(0.5, 1.32, ".", transform=ax.transAxes,
        ha="right", va="bottom", color="#fafafa",fontproperties=fontprop5 )
    ax.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=3,fancybox=True, prop=fontprop2,frameon=False)

    fig2 = plt.gcf()
    fig2.savefig(string2+'.png',dpi=100)



class req:
    def __init__(self,page=0, perPage=1000,baseDate="2020-02-26",sido="전국", returnType="XML"):
        query = {'serviceKey': ServiceKey, 'page':page,'perPage':perPage,'cond[sido::EQ]':sido,'cond[baseDate::GTE]':baseDate,'returnType':returnType}
        self.result = get(url=url,params=query)

a = req(returnType="JSON")
results = json.loads(a.result.text)
df0 = pd.read_csv("https://pastebin.com/raw/qUR2XWFu")
df = pd.DataFrame(results['data'], columns=['accumulatedFirstCnt','accumulatedSecondCnt', 'firstCnt','secondCnt','totalFirstCnt','totalSecondCnt'])
df = pd.concat([df0,df],ignore_index=True)
PopRatioOneDose = np.array(df['totalFirstCnt'].astype(int).tolist()) / PopOfSK
df['PopRatio-OneDose'] = PopRatioOneDose.tolist()
PopRatioFullyVaccinated = np.array(df['totalSecondCnt'].astype(int).tolist()) / PopOfSK
df['PopRatio-FullyVaccinated'] = PopRatioFullyVaccinated.tolist()
now = dt.datetime.now()
nHour = (now).strftime("%Y-%m-%d") if now.hour<10 else (now+dt.timedelta(days=1)).strftime("%Y-%m-%d")
nTime = now-dt.timedelta(days=1) if now.hour<10 else now
sp = np.datetime_as_string(np.arange('2020-02-26',nHour ,dtype='datetime64[D]'),unit='D')
#################
GenPlot(df.copy())