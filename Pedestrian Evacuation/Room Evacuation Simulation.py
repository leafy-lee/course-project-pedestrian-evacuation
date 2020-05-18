# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 09:57:35 2019

@author: asus
"""
from tkinter import *
import tkinter.messagebox as mb  
import pygame,sys
from pygame.locals import *
from tkinter import ttk
import numpy as np
import random
import math
import itertools

#1、设置场景模式所需参数字典
scene={'hall':[12,47,[1.6,1.6,1.6,1.6,0.8,0.8,0.8,0.8,1.6,1.6,1.6,1.6],[[0, 6],[0, 7],[0, 8],[0, 9],[3, 50],[4, 50],[11, 0],[12, 0],[15,43],[15, 44],[15, 45],[15, 46]],[[5, 8], [5, 9], [5, 10], [5, 11], [5, 12], [5, 13], [5, 33], [5, 34], [5, 35], [5, 36], [5, 37], [5, 38], [6, 8], [6, 9], [6, 10], [6, 11], [6, 12], [6, 13], [6, 33], [6, 34], [6, 35], [6, 36], [6, 37], [6, 38], [7, 8], [7, 9], [7, 10], [7, 11], [7, 12], [7, 13], [7, 33], [7, 34], [7, 35], [7, 36], [7, 37], [7, 38], [8, 8], [8, 9], [8, 10], [8, 11], [8, 12], [8, 13], [8, 33], [8, 34], [8, 35], [8, 36], [8, 37], [8, 38], [9, 8], [9, 9], [9, 10], [9, 11], [9, 12], [9, 13], [9, 33], [9, 34], [9, 35], [9, 36], [9, 37], [9, 38],[1, 1], [1, 2], [1, 3], [1, 4], [1, 5],[1, 10], [1, 11], [1, 12], [1, 13], [1, 14], [1, 15], [1, 16], [1, 17], [1, 18], [1, 19], [1, 20], [1, 21], [1, 22], [1, 23], [1, 24], [1, 25], [1, 26], [1, 27], [1, 28], [1, 29], [1, 30], [1, 31], [1, 32], [1, 33], [1, 34], [1, 35], [1, 36], [1, 37], [1, 38], [1, 39], [1, 40], [1, 41], [1, 42], [1, 43], [1, 44], [1, 45], [1, 46], [1, 47], [1, 48], [1, 49],[2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1], [8, 1], [9, 1], [10, 1],[13,1],[2,49],[5, 49], [6, 49], [7, 49], [8, 49], [9, 49], [10, 49], [11, 49], [12, 49], [13, 49], [14, 49],[14,48],[14,47],[14, 1], [14, 2], [14, 3], [14, 4], [14, 5], [14, 6], [14, 7], [14, 8], [14, 9], [14, 10], [14, 11], [14, 12], [14, 13], [14, 14], [14, 15], [14, 16], [14, 17], [14, 18], [14, 19], [14, 20], [14, 21], [14, 22], [14, 23], [14, 24], [14, 25], [14, 26], [14, 27], [14, 28], [14, 29], [14, 30], [14, 31], [14, 32], [14, 33], [14, 34], [14, 35], [14, 36], [14, 37], [14, 38], [14, 39], [14, 40], [14, 41], [14, 42]],[[[5,9],[8,13]],[[5,9],[33,38]],[[1,1],[1,5]],[[1,1],[10,49]],[[2,10],[1,1]],[[13,14],[1,1]],[[14,14],[2,42]],[[14,14],[47,49]],[[2,2],[49,49]],[[5,13],[49,49]]]],
       'little room':[8,20,[0.8,0.8],[[0, 5], [0, 6]],[[1,7],[1,8],[1, 1], [1, 2], [1, 3], [1, 4],[1, 9], [1, 10],[1, 11], [1, 12], [1, 13], [1, 14], [1, 15], [1, 16], [1, 17], [1, 18], [1, 19], [1, 20], [1, 21], [1, 22],[2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1], [8, 1], [9, 1], [10, 1],[2, 22], [3, 22], [4, 22], [5, 22], [6, 22], [7, 22], [8, 22], [9, 22], [10, 22],[10, 2], [10, 3], [10, 4], [10, 5], [10, 6], [10, 7], [10, 8], [10, 9],[10, 10], [10, 11], [10, 12], [10, 13], [10, 14], [10, 15], [10, 16], [10, 17], [10, 18], [10, 19], [10, 20], [10, 21]],[[[1,1],[1,4]],[[1,1],[7,22]],[[2,10],[1,1]],[[2,10],[22,22]],[[10,10],[2,21]]]]}

#2、用A*算法寻路
class AStar:   #实现A*算法
    class Node:  # 描述AStar算法中的节点数据
        def __init__(self, point, endPoint, g=0):
            self.point = point  # 自己的坐标
            self.father = None  # 上一个节点
            self.g = g  # g值，g值在用到的时候会重新算
            self.h = (abs(endPoint.x - point.x) + abs(endPoint.y - point.y)) * 10  # 计算h值

    def __init__(self, map2d, startPoint, endPoint, passTag=0):
        # 开启表
        self.openList = []
        # 关闭表
        self.closeList = []
        # 寻路地图
        self.map2d = map2d
        # 起点终点
        self.startPoint = startPoint
        self.endPoint = endPoint
        # 可行走标记
        self.passTag = passTag

    def findmin(self):
        currentNode = self.openList[0]
        for node in self.openList:
            if node.g + node.h < currentNode.g + currentNode.h:
                currentNode = node
        return currentNode

    def close(self, point):

        for node in self.closeList:
            if node.point == point:
                return True
        return False

    def open(self, point):

        for node in self.openList:
            if node.point == point:
                return node
        return None

    def endClose(self):
        for node in self.openList:
            if node.point == self.endPoint:
                return node
        return None

    def single_layer_search(self, minF, offsetX, offsetY):
        # 检测是否跨界
        if minF.point.x + offsetX < 0 or minF.point.x + offsetX > len(map2d) - 1 or minF.point.y + offsetY < 0 or minF.point.y + offsetY > len(map2d[0]) - 1:
            return
        # 如果是障碍，就忽略
        if self.map2d[minF.point.x + offsetX][minF.point.y + offsetY] != self.passTag:
            return
        # 如果在关闭表中，就忽略
        if self.close(Point(minF.point.x + offsetX, minF.point.y + offsetY)):
            return
        # 设置单位花费
        if offsetX == 0 or offsetY == 0:
            step = 10
        else:
            step = 14
        # 如果不在openList中，就把它加入openlist
        currentNode = self.open(Point(minF.point.x + offsetX, minF.point.y + offsetY))
        if not currentNode:
            currentNode = AStar.Node(Point(minF.point.x + offsetX, minF.point.y + offsetY), self.endPoint,
                                     g=minF.g + step)
            currentNode.father = minF
            self.openList.append(currentNode)
            return
        # 如果在openList中，判断minF到当前点的G是否更小
        if minF.g + step < currentNode.g:  # 如果更小，就重新计算g值，并且改变father
            currentNode.g = minF.g + step
            currentNode.father = minF

    def setNearOnce(self, x, y):
        offset = 1
        points = [[-offset, offset], [0, offset], [offset, offset], [-offset, 0],
                  [offset, 0], [-offset, -offset], [0, -offset], [offset, -offset]]
        for point in points:
            if 0 <= x + point[0] < self.map2d.w and 0 <= y + point[1] < self.map2d.h:
                self.map2d.data[x + point[0]][y + point[1]] = 1

    def start(self):
        # 1.将起点放入开启列表
        startNode = AStar.Node(self.startPoint, self.endPoint)
        self.openList.append(startNode)
        # 2.主循环逻辑
        while True:
            # 找到F值最小的点
            minF = self.findmin()
            # 把这个点加入closeList中，并且在openList中删除它
            self.closeList.append(minF)
            self.openList.remove(minF)
            # 判断这个节点的上下左右节点

            self.single_layer_search(minF, -1, 1)
            self.single_layer_search(minF, 0, 1)
            self.single_layer_search(minF, 1, 1)
            self.single_layer_search(minF, -1, 0)
            self.single_layer_search(minF, 1, 0)
            self.single_layer_search(minF, -1, -1)
            self.single_layer_search(minF, 0, -1)
            self.single_layer_search(minF, 1, -1)
            # 判断是否终止
            point = self.endClose()
            if point:  # 如果终点在关闭表中，就返回结果
                cPoint = point
                pathList = []
                while True:
                    if cPoint.father:
                        pathList.append(cPoint.point)
                        cPoint = cPoint.father
                    else:
                        # print(list(reversed(pathList)))
                        # print(pathList.reverse())
                        return list(reversed(pathList))
            if len(self.openList) == 0:
                return None

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __eq__(self, other):
        if self.x == other.x and self.y == other.y:
            return True
        return False
    def __str__(self):
        # return "x:"+str(self.x)+",y:"+str(self.y)
        return '(x:{}, y:{})'.format(self.x, self.y)
   
map2d=[]
def route(length,width,l0_discrete,lexit,lbarrier):   #寻路函数
    global scene
    #根据场景判断要走的门
    if length == 12:
        men=[[0, 6], [0, 7], [0, 8], [0, 9], [1, 6], [1, 7], [1, 8], [1, 9], [3, 49], [3, 50], [4, 49], [4, 50], [11, 0], [11, 1], [12, 0], [12, 1], [14, 43], [14, 44], [14, 45], [14, 46], [15, 43], [15, 44], [15, 45], [15, 46]]
    elif length == 8:
        men=[[0, 5], [0, 6]]    
    #记录结果
    RUOK=[]
    global map2d
    #m is the row while n is the column
    map2d=[[1 if i%(width+3)==0 else 0 for i in range(width+4)] if j%(length+3)!=0 else [1]*(width+4) for j in range(length+4)]
    #设置好场景里的空地障碍物和门
    for i in lbarrier:
        map2d[i[0]][i[1]]=1
    for i in men:
        map2d[i[0]][i[1]]=0
    #对列表里的每个人逐一求结果
    for i in range(len(l0_discrete)):
        #we need i has both the person's location and the nearest door's location
        a=l0_discrete[i]
        b=lexit[i]
        pStart, pEnd = Point(a[0],a[1]), Point(b[0], b[1])
        aStar = AStar(map2d, pStart, pEnd)
        pathList = aStar.start()
        point = pathList[0]
        dx,dy=point.x-pStart.x,point.y-pStart.y
        x,y=dx/(dx**2+dy**2)**0.5,dy/(dx**2+dy**2)**0.5
        RUOK.append([x,y])
    return RUOK

#3、用社会力模型模拟疏散
def g(x):
    if x<0:
        return 0
    else:
        return x  

def door(v,l0,lexit0,doorw,lexit0_ex):     #寻门函数
    if t4Chosen.get()=='nearest':  #最近出口
        lexit=[]
        for i in l0:
            pe=[]
            for j in lexit0_ex:
                pe.append(abs(i[0]-j[0])+abs(i[1]-j[1]))
            lexit.append(lexit0[pe.index(min(pe))])  
        return lexit    #生成出口列表
    else:    #最快出口
        dic_exit,all_exit={},{}
        for i in l0:
            pe1=[]
            for j in lexit0_ex:
                pe1.append(abs(i[0]-j[0])+abs(i[1]-j[1]))
            dic_exit[str(i)]=[pe1.index(min(pe1)),min(pe1)]
            all_exit[str(i)]=pe1
        sort_exit=[]
        for i in range(len(lexit0)):
            sort1=sorted(dic_exit.items(),key=lambda dic_exit:dic_exit[1][1])
            sort1=[j for j in sort1 if j[1]==i]
            sort1=[i[0] for i in sort1]
            sort_exit.append(sort1)    #先给每个人按距离远近分配出口
        lexit=[]
        for i in l0:
            pe2=[]
            for j in range(len(lexit0)):
                sort1=sort_exit[j]
                if str(i) in sort1:
                    n=sort1.index(str(i))
                else:
                    n=len(sort1)   #确定每个人要去某个门时前面的竞争人数
                pe2.append(all_exit[str(i)][j]/v+n/doorw[j])
            lexit.append(lexit0[pe2.index(min(pe2))])
        return lexit  #生成出口列表
        
    
def simulate():   #疏散模拟函数
#m0为质量列表，l0为初始位置列表，n为人数，t为单位时间，lexit为出口列表，lbarrier为障碍物占格列表，barrier为障碍物块状位置列表
    n,v=int(t2.get()),eval(t3.get())
    t=0.008/v
    length,width,doorw,lexit0,lbarrier,barrier=scene[t1Chosen.get()]
    v0=[[0,0] for i in range(n)]  #设初值
    m0=[random.uniform(45, 80) for i in range(n)] #随机生成行人质量
    #print(m0)
    #m0=[54.24427536501176, 59.121965100374055, 74.69413443968145, 48.79187391275063, 59.618480728251555, 51.339261942587186, 51.74346927797183, 73.83542686161958, 66.03644465627457, 48.81140836789046, 54.74277280148718, 48.532063796192794, 58.311529506749956, 71.53320990068545, 50.96109755822358, 69.34917067271118, 58.84840075807077, 72.90126825623916, 52.45004932819464, 65.22754636203803, 56.62983690976274, 56.773972989607884, 62.84122606993479, 49.371415636339805, 61.631886868988374, 67.96521138269694, 59.96022754848654, 67.05585464221471, 77.7751137619502, 59.1947957399119, 71.39660834713949, 45.80747210031694, 76.49626193653305, 56.822945801509256, 79.60248518690497, 76.63908503785264, 74.42214724239398, 74.68626191752433, 76.78179976075484, 74.92349746790647, 74.17773522097251, 72.31322886710191, 50.89185761809771, 53.25834249135557, 74.98063489672492, 67.6815887228876, 54.71953645141011, 59.79281739070662, 62.52673845902103, 56.94264534996185, 45.61056489006897, 51.683966644807604, 46.83564602948457, 68.41704087171048, 58.006007327370654, 66.2958353736121, 56.00496409485309, 76.82360144647933, 58.055190093165336, 47.31845953251094, 60.98654967321046, 71.34819089804623, 48.01524917864822, 62.92233283746794, 73.16751357970188, 78.59249075291498, 55.18396039759435, 59.85852114854886, 70.02138885298069, 68.71127139092164, 72.43561680151035, 75.44341164149452, 72.02467924330242, 53.259127791293864, 71.06712413431457, 78.06450212294364, 47.70886694126512, 79.2499226820886, 51.451174885798544, 65.14819178630968, 76.46985671516256, 50.41633245011515, 45.10130546902471, 63.94062558741946, 73.85140440992204, 77.79948821326869, 61.1707821913409, 64.75559628613598, 59.40873805508758, 62.37080434362753, 69.09831517507011, 67.89084716172886, 77.44088855065942, 54.33137542282273, 47.549805112186384, 67.6729174494652, 69.4240454992518, 65.27663810732892, 47.69666920327779, 52.776737816650225]
    l0=list(itertools.product(range(2,2+length), range(2,2+width)))  #生成房间内部的所有格子
    l0=[list(i) for i in l0 if list(i) not in lbarrier]  #将有障碍物的格子去掉
    l0=random.sample(l0,n) #在l0中随机产生n个不同的格子位置
    #print(l0)
    #l0=[[10, 38], [5, 21], [5, 24], [9, 22], [7, 17], [11, 42], [2, 24], [2, 7], [12, 43], [11, 46], [12, 35], [8, 23], [5, 15], [9, 5], [3, 3], [12, 11], [13, 27], [2, 36], [12, 19], [11, 21], [11, 24], [7, 25], [8, 45], [10, 17], [9, 19], [10, 35], [11, 39], [13, 7], [5, 5], [5, 41], [2, 28], [10, 48], [9, 16], [13, 24], [3, 47], [8, 6], [9, 32], [4, 40], [13, 4], [13, 14], [11, 48], [6, 16], [8, 17], [2, 4], [6, 47], [10, 32], [8, 42], [11, 20], [13, 41], [12, 45], [4, 28], [4, 32], [12, 12], [2, 16], [7, 26], [8, 5], [11, 13], [10, 33], [9, 40], [9, 4], [4, 23], [5, 30], [6, 23], [7, 14], [8, 32], [6, 28], [6, 20], [13, 31], [12, 2], [3, 31], [10, 26], [13, 22], [12, 34], [7, 40], [8, 2], [13, 37], [7, 42], [11, 22], [10, 18], [12, 21], [2, 31], [2, 18], [4, 24], [10, 8], [4, 30], [4, 37], [9, 20], [8, 27], [8, 15], [13, 17], [2, 37], [11, 14], [11, 33], [3, 12], [3, 40], [5, 23], [3, 48], [7, 46], [10, 27], [12, 33]]
    l0=[list(np.array(i)*0.4+0.2) for i in l0]  #将生成的随机位置调整为实际距离
    lexit0_ex=[list(np.array(i)*0.4+0.2) for i in lexit0]
    lexit=door(v,l0,lexit0,doorw,lexit0_ex) #调用函数得到出口列表 
    #print(l0)
    #以下绘制疏散动图
    pygame.init() # 实始化pygame引擎    
    screen=pygame.display.set_mode((200, int(500*width/40)), pygame.RESIZABLE,32)# 创建屏幕，它是一个图层    
    pygame.display.set_caption('evacuation process simulation')# 设置窗口标题
    passenger=pygame.image.load('passenger1.png')
    passenger=pygame.transform.scale(passenger,(6, 6))
    clock = pygame.time.Clock()
    screen.fill((255, 255, 255))  #设置开始为全白
    for i in l0:
        screen.blit(passenger,(i[0]*15+7,i[1]*15+7))  #画行人
    for i in barrier:   #画墙和柱子
        pygame.draw.rect(screen,(0,0,255),(i[0][0]*6+10,i[1][0]*6+10,(i[0][1]-i[0][0]+1)*6,(i[1][1]-i[1][0]+1)*6),0)
    clock.tick(80)
    pygame.display.update()  #更新画布
    s=0
    #lv1,lv2,lv3,lv4,lv5,lv6=[],[],[],[],[],[]  #便于画速度密度图时得知不同密度下的速度
    while len(l0)>0:
        pygame.display.update()
        a,v1,l1=[],[],[]
        l0_discrete=[[int(i[0]/0.4),int(i[1]/0.4)] for i in l0]
        if t4Chosen.get()=='fastest':
            lexit=door(v,l0,lexit0,doorw,lexit0_ex)  #如果选择优化的最快出口模式，需要每次循环都更新出口
        e=route(length,width,l0_discrete,lexit,lbarrier)  #调用寻路函数得到意欲走的方向
        screen.fill((255, 255, 255))  #将画布设置为全白
        for i in range(len(l0)):
            near,corner_dic={},{}  #计算相距不超过3m的障碍物和行人对某人的作用力
            for j in range(len(l0)):
                if j!=i:                   
                    ri_rj=np.array(l0[i])-np.array(l0[j])
                    dij=np.sqrt(np.sum(np.square(ri_rj)))               
                    #if dij<=min(3,width/4,length/4):
                    if dij<=3:
                        tij=list(ri_rj)[:]
                        tij.reverse()
                        tij[0]=-tij[0]
                        tij=np.array(tij)/dij
                        vji=np.array(v0[j])-np.array(v0[i])
                        vj_vit=np.sum(vji*tij)
                        delta=0.4-dij
                        g0=g(delta)
                        fij=(2000*math.exp(delta/0.08)+1.2*(10**5)*g0)*ri_rj/dij+2.4*(10**5)*g0*vj_vit*tij
                        near[str(j)+'r']=[fij,dij-0.4]
            for j in range(len(barrier)):
                x1,x2,y1,y2=barrier[j][0][0]*0.4,(barrier[j][0][1]+1)*0.4,barrier[j][1][0]*0.4,(barrier[j][1][1]+1)*0.4
                if x1<=l0[i][0]<=x2:
                    d1,d2=abs(l0[i][1]-y1),abs(l0[i][1]-y2)
                    if d1>d2:
                        dij,ri_rj,tij=d2,np.array([0,1]),np.array([-1,0])
                    else:
                        dij,ri_rj,tij=d1,np.array([0,-1]),np.array([1,0])
                    #if dij<=min(3,width/4,length/4):
                    if dij<=3:
                        vji=-np.array(v0[i])
                        vj_vit=np.sum(vji*tij)
                        delta=0.2-dij
                        g0=g(delta)
                        fij=(2000*math.exp(delta/0.08)+1.2*(10**5)*g0)*ri_rj+2.4*(10**5)*g0*vj_vit*tij
                        near[j]=[fij,dij-0.2]
                    if x2-x1==0.4:
                        near[j]=[fij,dij-0.2]
                elif y1<=l0[i][1]<=y2:
                    d1,d2=abs(l0[i][0]-x1),abs(l0[i][0]-x2)
                    if d1>d2:
                        dij,ri_rj,tij=d2,np.array([1,0]),np.array([0,1])
                    else:
                        dij,ri_rj,tij=d1,np.array([-1,0]),np.array([0,-1])
                    #if dij<=min(3,width/4,length/4):
                    if dij<=3:
                        vji=-np.array(v0[i])
                        vj_vit=np.sum(vji*tij)
                        delta=0.2-dij
                        g0=g(delta)
                        fij=(2000*math.exp(delta/0.08)+1.2*(10**5)*g0)*ri_rj+2.4*(10**5)*g0*vj_vit*tij
                        near[j]=[fij,dij-0.2]
                    if y2-y1==0.4:
                        near[j]=[fij,dij-0.2]
                else:
                    d1,d2,d3,d4=((l0[i][0]-x1)**2+(l0[i][1]-y1)**2),((l0[i][0]-x2)**2+(l0[i][1]-y1)**2),((l0[i][0]-x1)**2+(l0[i][1]-y2)**2),((l0[i][0]-x2)**2+(l0[i][1]-y2)**2)
                    dij=min(d1,d2,d3,d4)
                    #if dij<=(min(3,width/4,length/4))**2:
                    if dij<=9:
                        corner={d1:[x1,y1],d2:[x2,y1],d3:[x1,y2],d4:[x2,y2]}
                        ri_rj=np.array(l0[i])-np.array(corner[dij])
                        dij=dij**0.5
                        tij=list(ri_rj)[:]
                        tij.reverse()
                        tij[0]=-tij[0]
                        tij=np.array(tij)/dij
                        vji=-np.array(v0[i])
                        vj_vit=np.sum(vji*tij)
                        delta=0.2-dij
                        g0=g(delta)
                        fij=(2000*math.exp(delta/0.08)+1.2*(10**5)*g0)*ri_rj/dij+2.4*(10**5)*g0*vj_vit*tij
                        corner_dic[j]=[fij,dij-0.2]
                
            a_self=(np.array(e[i])*v-np.array(v0[i]))/0.5  #计算本人想出门产生的加速度
            a_t=list(a_self)
            a_t.reverse()
            a_t[0]=-a_t[0]
            a_t=np.array(a_t)   #计算加速度的法向量
            e_t=e[i][::-1]
            e_t[0]=-e_t[0]
            e_t=np.array(e_t)  #计算原本行进方向的法向量
            #a_d=np.array(list(a_self)[:])  #处理静止问题的一个尝试
            a_t_unit=a_t/np.sqrt(np.sum(np.square(a_t))) #计算加速度的法向量的单位向量
            #ld_comity=[0.2*t]  #礼让模型：计算后退距离的列表
            for j in near:
                a_self+=near[j][0]/m0[i] 
                #ld_comity.append(near[j][1]/0.9) #礼让模型：计算所有障碍物的距离*0.9
                '''if np.sum(near[j][0]*a_t)==0:
                    a_d+=near[j][0]/m0[i]   #处理静止问题的尝试，记录与加速度平行的外作用力'''
            for j in corner_dic:
                if s>=2:
                    a_self+=np.sum(a_t_unit*corner_dic[j][0])*a_t_unit/m0[i]
                else:
                    a_self+=corner_dic[j][0]/m0[i]  #开始先按原模型计算，以免开始的随机位置正好靠墙被瞬间撞飞。后面用优化模型的算法计算
                #ld_comity.append(corner_dic[j][1]/0.9) #礼让模型：计算所有障碍物的距离*0.9
                '''if np.sum(corner_dic[j][0]*a_t)==0:
                    a_d+=corner_dic[j][0]/m0[i]   #处理静止问题的尝试，记录与加速度平行的外作用力'''
            '''if list(a_self)==[0,0]:
                a_self=a_d   #处理静止问题的尝试,如果受力平衡，就只算与加速度平行的力'''
            a.append(list(a_self))
            v_self=np.array(v0[i])+a_self*t
            #v_t=list(np.array(v0[i])*v_self)  #礼让模型：计算更新速度与原速度的矢量点乘
            v1.append(list(v_self))
            location=np.array(l0[i])+v_self*t
            '''if len(l0)<=10:    
                if (v_t[0]*v0[i][0]<0) and (v_t[1]*v0[i][1]<0):
                    v1.append([0,0])
                    d_standard=np.sqrt(np.sum(np.square(v_self*t)))
                    d_comity=min(ld_comity)
                    d_retreat=max(d_standard,d_comity)
                    location=np.array(l0[i])+v_self*t*d_retreat/d_standard  #礼让模型：若徘徊则后退一步，打破僵局 '''                 
            l1.append(list(location))
            screen.blit(passenger,tuple(15*(location)+7))

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        for i in barrier:
            pygame.draw.rect(screen,(0,0,255),(i[0][0]*6+10,i[1][0]*6+10,(i[0][1]-i[0][0]+1)*6,(i[1][1]-i[1][0]+1)*6),0)
        clock.tick(80)
        pygame.display.update()  #绘图，更新画布
        '''lv=[]
        for i in range(len(l1)):
            if (1<=int(l1[i][0]/0.4)<=2) and (5<=int(l1[i][1]/0.4)<=6):
               lv.append('%.2f'%np.sqrt(np.sum(np.square(np.array(v1[i])))))
        num=len(lv)
        lv=[eval(i) for i in lv]
        if num==1:
            lv1.append(lv[0])
        if num==2:
            lv2.append(sum(lv)/2)
        if num==3:
            lv3.append(sum(lv)/3)
        if num==4:
            lv4.append(sum(lv)/4)
        if num==5:
            lv5.append(sum(lv)/5)
        if num==6:
            lv6.append(sum(lv)/6)   #得到绘制速度密度图所需数据'''
        l_copy=l1[:]
        for i in range(len(l_copy)-1,-1,-1):
            if (l_copy[i][0]<0.4) or (l_copy[i][0]>(3+length)*0.4) or (l_copy[i][1]<0.4) or (l_copy[i][1]>(3+width)*0.4):
                m0.pop(i)
                l1.pop(i)
                v1.pop(i)
                lexit.pop(i)  #将出门的行人从各列表移除
        v0,l0,s=v1,l1,s+1   #重置初始列表
    mb.showinfo('result','The evacuation time is '+'%.2f'%(s*t)+'s')  #返回疏散时间
    '''print(lv1)
    print('==============')
    print(lv2)
    print('==============')
    print(lv3)
    print('==============')
    print(lv4)
    print('==============')
    print(lv5)
    print('==============')
    print(lv6)
    print('==============')   #得到绘制速度密度图所需数据 '''
        
#4.设置主窗口主要部件并排版。
top=Tk()
top.title('Room Evacuation Simulation')
top.geometry('200x400')
fr1=Frame(top)
#bg1=PhotoImage(file="bg21.png")
#bg2=PhotoImage(file="bg31.png")
l1=Label(fr1,text='SCENE MODE',font=("华文隶书",12),anchor="e") 
t1=StringVar()   
t1Chosen=ttk.Combobox(fr1,width=15,textvariable=t1,state='readonly') 
t1Chosen['values']=('hall','little room')
t1Chosen.current(0)
l2=Label(fr1,text='CROWD (/person)',font=("华文隶书",12),compound=CENTER)  
t2=Entry(fr1,width=18)
l3=Label(fr1,text='SPEED (m/s)',font=("华文隶书",12),compound=CENTER)  
t3=Entry(fr1,width=18)
l4=Label(fr1,text='EXIT JUDGEMENT',font=("华文隶书",12),compound=CENTER)  
t4=StringVar()   
t4Chosen=ttk.Combobox(fr1,width=15,textvariable=t4,state='readonly') 
t4Chosen['values']=('nearest','fastest')
t4Chosen.current(0)
b1=Button(fr1,text="SIMULATE",command=simulate) 
fr1.grid(row=0,column=0,columnspan=1,padx=20,pady=15)
l1.grid(row=0,column=0,sticky=W,pady=8)
t1Chosen.grid(row=1,column=0,sticky=W,pady=8)
l2.grid(row=2,column=0,sticky=W,pady=8)
t2.grid(row=3,column=0,sticky=W,pady=8)
l3.grid(row=4,column=0,sticky=W,pady=8)
t3.grid(row=5,column=0,sticky=W,pady=8)
l4.grid(row=6,column=0,sticky=W,pady=8)
t4Chosen.grid(row=7,column=0,sticky=W,pady=8)
b1.grid(row=8,column=0,pady=15) 
mainloop()
