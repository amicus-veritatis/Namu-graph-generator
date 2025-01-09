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
ServiceKey = #############################
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
path = '#############################\\NotoSansCJKkr-Black.otf'
path2 = '#############################\\NotoSansCJKkr-Bold.otf'
path3 = '#############################\\NotoSansCJKkr-Medium.otf'
path4 = '#############################\\NotoSansCJKkr-Regular.otf'
fontprop = fm.FontProperties(fname=path)
fontprop3 = fm.FontProperties(fname=path2, size=145)
fontprop2 = fm.FontProperties(fname=path, size=64)  # legend
fontprop4 = fm.FontProperties(fname=path2, size=48)
fontprop4 = fm.FontProperties(fname=path2, size=72)
fontprop5 = fm.FontProperties(fname=path3, size=54)
fontprop6 = fm.FontProperties(fname=path, size=48)
fontprop7 = fm.FontProperties(fname=path, size=40)
fontprop8 = fm.FontProperties(fname=path, size=88)
font1 = fontprop.get_name()
font2 = fontprop2.get_name()
matplotlib.rcParams["font.family"] = font2


# matplotlib.rcParams.rc('font', family='Malgun Gothic')

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
    Axis.axvspan(Date[0], Date[1], facecolor=Color, alpha=Alpha)


def LineEmphasizer(Axis=None, Date=None, Data=None, Color=None, LineWidth=None, Label=None):
    if not all(x is not None for x in (Axis, Date, Color, LineWidth)):
        raise ValueError("Every values has to be filled\n")
    elif Data.size == None:
        raise ValueError("Data가 비어있습니다.\n")
    Axis.plot(Date, Data.astype(int).tolist(), color=Color, linewidth=LineWidth, label=Label)


def millions(x, pos):
    """The two args are the value and tick position."""
    return '{:1.1f}M'.format(x * 1e-6)


def mans(x, pos):
    """The two args are the value and tick position."""
    if x == 0:
        return 0
    else:
        return '{:1.0f}만'.format(x * 1e-4)


def PeriodLocator(ax, start=None, end=None):
    if all(x is not None for x in (start, end)):
        locs = [mdates.date2num(start)] + list(ax.get_xticks()) + [mdates.date2num(end)]
    elif start is None and end is None:
        locs = list(ax.get_xticks())
    elif start is None:
        locs = list(ax.get_xticks()) + [mdates.date2num(end)]
    elif end is None:
        locs = [mdates.date2num(start)] + list(ax.get_xticks())

    locator = matplotlib.ticker.FixedLocator(locs)
    return locator


def TickAdjust(ax=None, tickAxis="Y", recent=None, Min=0):
    """
    425만 -> 500만 같이 Data의 lim을 변화시켜 Tick을 조정시키는 함수
    log10(a*10^M+b*10^(M-1)....) = M + log10(a+b*10^(-1)+....)
    """
    M = math.log10(recent)
    A = math.pow(10, M - math.floor(M))
    A = math.ceil(A)
    M = math.floor(M)
    Max = A * math.pow(10, M)

    print("Max : " + str(Max))
    
    if recent<Max*0.75:
        Max = min(math.floor(((Max*0.75) / 5000000)+1),math.ceil(((Max*0.75) / 5000000)))*5000000
        print("Max : " + str(Max))
        
    lim_range = [Min, Max]
    if tickAxis == "Y":
        ax.set_ylim(lim_range)
    elif tickAxis == "X":
        ax.set_xlim(lim_range)
    else:
        raise (ValueError("tickAxis is neither Y nor X"))


def choose_fill(ax=None, DataIn=None, days=None, FirstDose=True, fColor="#FBB4B1", SecondDose=True, sColor="#e3120b",
                TotalDose=True, tColor="#cad9e1", LineWidth=3):
    if not all(x is not None for x in (ax, DataIn,days)):
        raise ValueError("ax and DataIn must be specified\n")

    if FirstDose and SecondDose and TotalDose:
        ax.fill_between(days, DataIn['totalFirstCnt'].astype(int).tolist(), color=fColor, label="1차 접종자")
        ax.fill_between(days, DataIn['totalSecondCnt'].astype(int).tolist(), color=sColor, label="접종 완료자")
        ax.fill_between(days, (np.array(DataIn['totalSecondCnt'].astype(int).tolist()) + np.array(
            DataIn['totalFirstCnt'].astype(int).tolist())).tolist(), color=tColor, label="총 접종건수")
        ax.fill_between(days, DataIn['totalFirstCnt'].astype(int).tolist(), color=fColor)
        ax.fill_between(days, DataIn['totalSecondCnt'].astype(int).tolist(), color=sColor)
        LineEmphasizer(ax, days, DataIn['totalFirstCnt'], '#fafafa', LineWidth)
        LineEmphasizer(ax, days, DataIn['totalSecondCnt'], '#fafafa', LineWidth)
    elif FirstDose and SecondDose and not TotalDose:
        ax.fill_between(days, DataIn['totalFirstCnt'].astype(int).tolist(), color=fColor, label="1차 접종자")
        ax.fill_between(days, DataIn['totalSecondCnt'].astype(int).tolist(), color=sColor, label="접종 완료자")
        LineEmphasizer(ax, days, DataIn['totalFirstCnt'], '#fafafa', LineWidth)
        LineEmphasizer(ax, days, DataIn['totalSecondCnt'], '#fafafa', LineWidth)
    elif not TotalDose:
        if FirstDose:
            ax.fill_between(days, DataIn['totalFirstCnt'].astype(int).tolist(), color=fColor, label="1차 접종자")
        elif SecondDose:
            ax.fill_between(days, DataIn['totalSecondCnt'].astype(int).tolist(), color=sColor, label="접종 완료자")
    elif TotalDose:
        if FirstDose:
            ax.fill_between(days, DataIn['totalFirstCnt'].astype(int).tolist(), color=fColor, label="1차 접종자")
            ax.fill_between(days, (np.array(DataIn['totalSecondCnt'].astype(int).tolist()) + np.array(
                DataIn['totalFirstCnt'].astype(int).tolist())).tolist(), color=tColor, label="총 접종건수")
            ax.fill_between(days, DataIn['totalFirstCnt'].astype(int).tolist(), color=fColor)
            LineEmphasizer(ax, days, DataIn['totalFirstCnt'], '#fafafa', LineWidth)
        elif SecondDose:
            ax.fill_between(days, DataIn['totalSecondCnt'].astype(int).tolist(), color=sColor, label="접종 완료자")
            ax.fill_between(days, (np.array(DataIn['totalSecondCnt'].astype(int).tolist()) + np.array(
                DataIn['totalFirstCnt'].astype(int).tolist())).tolist(), color=tColor, label="총 접종건수")
            ax.fill_between(days, DataIn['totalSecondCnt'].astype(int).tolist(), color=sColor)
            LineEmphasizer(ax, days, DataIn['totalSecondCnt'], '#fafafa', LineWidth)


def GenPlot(DataIn):
    left = dt.date(2021, 2, 26)
    right = dt.date.today()
    tick_spacing = 5000
    nDay = right if now.hour < 10 else right + dt.timedelta(days=1)
    days = mdates.drange(left, nDay, dt.timedelta(days=1))
    days = days[:-1] if len(days) != len(DataIn.iloc[::]) else days
    print("len(DataIn.iloc[::]) : " + str(len(DataIn.iloc[::])) + "\n")
    fig, [ax, timeline] = plt.subplots(nrows=2, ncols=1, sharex='col',
                                       gridspec_kw={'height_ratios': [36, 0.001], 'wspace': 0, 'hspace': 0},
                                       figsize=(48, 32), constrained_layout=True, facecolor='#fafafa')  ### [36,20]
    params = {"figure.facecolor": "#fafafa",
              "axes.facecolor": "#fafafa",
              "axes.grid": True,
              "axes.grid.axis": "y",
              "grid.color": "#ffffff",
              "grid.linewidth": 10,
              "axes.spines.left": False,
              "axes.spines.right": False,
              "axes.spines.top": False,
              "ytick.major.size": 0,
              "ytick.minor.size": 0,
              "xtick.direction": "in",
              "xtick.major.size": 8,
              "xtick.color": "#191919",
              "axes.edgecolor": "#191919",
              "text.color": "#4a4a4a",
              "font.family": font1,
              'axes.labelcolor': "#4a4a4a",
              "axes.prop_cycle": plt.cycler('color',
                                            ['#006767', '#ff7f0e', '#2ca02c', '#d62728',
                                             '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                                             '#bcbd22', '#17becf'])}
    plt.rcParams.update(params)

    ######RatAx : 비율#######
    '''
        1차 : totalFirstCnt / PopOfSK * 100 %
        2차 : totalSecondCnt / PopOfSk * 100 %
    '''
    ax.set_title('대한민국 내 COVID-19 백신 접종 현황', fontproperties=fontprop3, pad=12, fontdict=dict(color="#191919"),
                 loc='left')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y\n%m'))

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.yaxis.set_major_formatter(mans)
    TickAdjust(ax=ax, tickAxis="Y", recent=DataIn['totalFirstCnt'].iloc[-1], Min=0)
    ###     TickAdjust(ax=ax, tickAxis="Y", recent=DataIn['totalSecondCnt'].iloc[-1] + DataIn['totalFirstCnt'].iloc[-1], Min=0)
    ###ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))
    ax.set_facecolor('#fafafa')
    ax.yaxis.tick_right()

    for label in ax.get_yticklabels():
        label.set_fontproperties(fontprop4)
        label.set_color('#4a4a4a')
        ######trans#######
    ax.grid(True, 'major', 'y', color='#B6B6B6', linestyle='-', linewidth=3, alpha=1)
    # ffffff 3E647D
    ax.set_axisbelow(True)
    choose_fill(ax, DataIn, days, TotalDose=False)
    '''
    LineEmphasizer(ax,days,np.array(DataIn['totalSecondCnt'].astype(int).tolist())+np.array(DataIn['totalFirstCnt'].astype(int).tolist()),'#fafafa',3)
    '''
    #######

    for axs in [ax, timeline]:
        axs.set_frame_on(False)
        ########### mv avg ################
        '''
    avg_dConf = mv_avg(DataIn['dConf'],7,"#823c5a") #e3120b
    ax2.plot(days, avg_dConf.sma().tolist(), color=avg_dConf.color, linewidth=8)
    '''
    ax.set_aspect('auto')
    ax.xaxis.set_label_position('top')
    string = "" + nTime.strftime("%Y/%m/%d") + " 기준"
    print(DataIn['PopRatio-OneDose'])
    FDRate = DataIn['PopRatio-OneDose'].iloc[-1] * 100
    print(FDRate)
    SDRate = DataIn['PopRatio-FullyVaccinated'].iloc[-1] * 100
    print(SDRate)
    FDRate = "%0.1f%%" % FDRate
    SDRate = "%0.1f%%" % SDRate
    print("%s, %s\n" % (FDRate, SDRate))
    RatioString = "인구 대비 접종률: 1차 접종 %s, 접종 완료 %s" % (FDRate, SDRate)
    string = string + "\n"
    ax.set_xlabel(string, fontproperties=fontprop4, labelpad=32, color="#4a4a4a", loc='left')
    ##### Timeline #########
    '''
    Vac = ['아스트라제네카','얀센','화이자','모더나','노화이자','모더나','노바백스']
    data = [    (dt.datetime(2021,02,14),dt.datetime(2021,02,26),'화이자'),
                (dt.datetime(2021,02,14),dt.datetime(2021,03,01),'아스트라제네카'),
                (dt.datetime(2021,03,01),dt.datetime(2021,04,01),'화이자'),
                (dt.datetime(2021,02,01),dt.datetime(2021,03,01),'화이자'),
                (dt.datetime(2021,02,01),dt.datetime(2021,03,01),'화이자'),
                (dt.datetime(2021,02,01),dt.datetime(2021,03,01),'화이자'),
                (dt.datetime(2021,02,01),dt.datetime(2021,03,01),'화이자'),
                (dt.datetime(2021,02,01),dt.datetime(2021,03,01),'화이자'),
                (dt.datetime(2021,02,01),dt.datetime(2021,03,01),'화이자'),

            ]
    
    txts = ['아스트라제네카 157.4만회\n화이자 11.7만회','요양병원·요양 시설\n입소자·종사자\n(65세 미만)', '코로나19 환자\n치료병원\n종사자', '고위험\n의료기관\n종사자', '코로나19\n1차\n대응요원', '화이자 100만회',

        ,,'']
    txts = list(reversed(txts))
    print(txts)
    dates = ['2021-03-25']
    dates = [dt.datetime.strptime(d, "%Y-%m-%d") for d in dates]
    dates.append(left)
    dates.insert(0,right)
    print(dates)
    levels=[0,1,0]
    hets=[]
    levels = np.flip(levels).tolist()
    print(levels)
    '''
    # 9일부터 주석 풀기
    timeline.set_xlim([left, right])
    timeline.tick_params(axis="x", labelsize=40)
    ##timeline.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    #    timeline.xaxis.set_major_locator(ticker.AutoLocator())
    timeline.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=FR, interval=2))
    timeline.xaxis.set_major_locator(PeriodLocator(timeline, start=None, end=None))
    #    timeline.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=FR,interval=2))
    timeline.tick_params(axis="x", which='major', length=30, width=0, labelsize=56)
    timeline.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=MO, interval=2))
    timeline.xaxis.set_major_formatter(mdates.DateFormatter('%Y\n%m/%d'))
    '''
    timeline.vlines(dates, 0, levels, color="#e3120b")
    timeline.plot(dates, np.zeros_like(dates), "-o",
        color="k", markerfacecolor="w", markersize=20)  # Baseline and markers on it.
        '''
    # annotate line
    '''
    ha = ["right","left","center"]
    ha = ["right","center","left","left","center","center","right","right","left","center"]
    he = itt.cycle(ha)
    for d, l, r in zip(dates, levels, txts):
      timeline.annotate(r, xy=(d, l), xytext=(-3, np.sign(l)*3), textcoords="offset points", verticalalignment="bottom" if l > 0 else "top", ha="center"if l > 0 else next(he), fontproperties=fontprop6 if l>0 else fontprop7)

    timeline.text(0, -0.3, "자료 : 질병관리청", transform=timeline.transAxes,
        ha="left", va="top", color="#4a4a4a",
        fontproperties=fontprop4)
    '''

    textbox = AnchoredText(RatioString, frameon=False, loc=2, pad=0,
                           prop=dict(fontproperties=fontprop4, color="#4a4a4a"))
    plt.setp(textbox.patch, facecolor='#fafafa', alpha=1)
    ax.add_artist(textbox)

    plt.annotate('자료 : 질병관리청', (0, 0), (0, -180), fontproperties=fontprop4,
                 xycoords='axes fraction', textcoords='offset points', va='top')
    timeline.set_ylim([-2, 2])
    timeline.yaxis.tick_left()
    timeline.get_yaxis().set_visible(False)  ############
    timeline.set_facecolor("#fafafa")

    right = (np.datetime64('today').astype('datetime64[M]') + np.timedelta64(5, 'W')).astype('datetime64[D]')

    mon = np.arange(np.datetime64('2021-02-21'), right, np.timedelta64(1, 'W'), dtype='datetime64[W]')
    mon = mon.tolist()

    for (i, j) in zip(mon[1::2], mon[2::2]):
        timeline.axvspan(i, j, facecolor='#FBB4B1', alpha=1)
    #   timeline.grid(True,'minor', 'x', color="#4a4a4a", alpha=0.5, linewidth=3)
    string = "대한민국 내 COVID-19 백신 접종 현황 ("
    Dtime = nTime.strftime("%m.%d")
    string2 = string + Dtime + " 기준)"

    ###
    timeline.set_aspect('auto')

    ###

    ax.text(0.5, 1.2, ".", transform=ax.transAxes,
            ha="right", va="center", color="#fafafa", fontproperties=fontprop5)
    timeline.text(0.5, -0.15, ".", transform=ax.transAxes,
                  ha="center", va="center", color="#fafafa", fontproperties=fontprop5)
    ax.legend(bbox_to_anchor=(1, 1.1), loc='upper right', ncol=3, fancybox=True, prop=fontprop2, frameon=False)

    fig2 = plt.gcf()
    fig2.savefig(string2 + '.png', dpi=100)


class req:
    def __init__(self, page=0, perPage=1000, baseDate="2020-02-26", sido="전국", returnType="XML"):
        query = {'serviceKey': ServiceKey, 'page': page, 'perPage': perPage, 'cond[sido::EQ]': sido,
                 'cond[baseDate::GTE]': baseDate, 'returnType': returnType}
        self.result = get(url=url, params=query)



a = req(returnType="JSON")
results = json.loads(a.result.text)
print(results['data'])
df0 = pd.read_csv("https://pastebin.com/raw/qUR2XWFu")
df = pd.DataFrame(results['data'],
                  columns=['accumulatedFirstCnt', 'accumulatedSecondCnt', 'firstCnt', 'secondCnt', 'totalFirstCnt',
                           'totalSecondCnt'])
df = pd.concat([df0, df], ignore_index=True)
print("DamedaDAme\n")
PopRatioOneDose = np.array(df['totalFirstCnt'].astype(int).tolist()) / PopOfSK
df['PopRatio-OneDose'] = PopRatioOneDose.tolist()
PopRatioFullyVaccinated = np.array(df['totalSecondCnt'].astype(int).tolist()) / PopOfSK
df['PopRatio-FullyVaccinated'] = PopRatioFullyVaccinated.tolist()
print(df)
now = dt.datetime.now()
print(now)  #########################
print(dt.datetime.hour)  ################
print(now.hour)  ################
nHour = (now).strftime("%Y-%m-%d") if now.hour < 10 else (now + dt.timedelta(days=1)).strftime("%Y-%m-%d")
nTime = now - dt.timedelta(days=1) if now.hour < 10 else now
sp = np.datetime_as_string(np.arange('2020-02-26', nHour, dtype='datetime64[D]'), unit='D')
#################
print(sp)
print(df['firstCnt'])
print(df['totalSecondCnt'])
df['totalSecondCnt'].iloc[143] = np.mean(np.array(df['totalSecondCnt'].iloc[142:145:2])).astype(int)
df['totalFirstCnt'].iloc[143] = np.mean(np.array(df['totalFirstCnt'].iloc[142:145:2])).astype(int)
key_colums = ['stateDt', 'decideCnt', 'clearCnt', 'dConf', 'Event', 'Eventline']

GenPlot(df.copy())

# dConfirm = np.append(np.array([1]),np.array(oConfirm))
# GenPlot(df[key_colums].copy())
# print(df[key_colums])
# for date, conf, rlse, dead, care, seq in zip(Datetime,Confirm,Release,Death_toll,CARE,Seq):
#    print('{0}\t\t{1}\t{2}\t{3}'.format(str(int(date.string)-1),conf.string,dead.string,rlse.string))
