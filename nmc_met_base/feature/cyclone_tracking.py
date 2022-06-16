# _*_ coding: utf-8 _*_

# Copyright (c) 2022 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
Lagrangian cyclone tracking algorithm developed while at the National Snow 
and Ice Data Center Inputs include several raster data sets: sea level pressure 
fields, a digital elevation model, arrays of latitude, longitude, and x and y 
distances across the input grid. A suite of parameters set by the user are also 
needed, and the algorithm assumes that the user has already 
a) set up some output directories and 
b) regridded all inputs to an equal-area grid. 
Polar grids (e.g., the EASE2 grid) are ideal. An example script is included in 
Version 12 Scripts that shows how ERA5 data were converted. 

Adapted from https://github.com/alexcrawford0927/cyclonetracking
"""

'''
Main Author: Alex Crawford
Contributions from: Mark Serreze, Nathan Sommer
Date Created: 20 Jan 2015 (original module)
Modifications: (since branch from version 11)
21 May 2020 --> Improved speed of kernelgradient function (courtesy Nathan Sommer) & laplacian function 
02 Jun 2020 --> Made test for weak minima mandatory during cyclone detection (removing an if statement)
            --> Re-ordered the unit conversions in the haversine formula to prioritize m and km
17 Jun 2020 --> Modified how the area mechanism for single-center cyclones works to make 
                it fewer lines of code but the same number of steps
            --> Minor tweak in cTrack2sTrack function so that pandas data.frame row 
                is pulled out as its own dataframe instead of always being subset
11 Sep 2020 --> Pulled the "maxdist" calculation out of the module and into the top-level
                script because it really only needs to be done once
            --> Changed method of determining track continuation during merges to 
                prioritize nearest neighbor, but then maximum depth (instead of lifespan)
02 Oct 2020 --> Changed the findAreas method of cyclonefield objects so that
                the fieldCenters is replaced instead of duplicated when distinguishing 
                primary and secondary centers. Also made area and center fields the uint8 data type.
                This greatly reduces the file size of detection outputs.
05 Oct 2020 --> Changed the parent track id assignment during area merges to also 
                prioritize depth instead of lifespan
            --> Added try/except statements to the cTrack2sTrack function so that it will ignore
                cases where tids between two months don't align insted of breaking.
Start Version 12_4 (Branch from 12_2)
13 Jan 2021 --> Removed the columns precip, preciparea, and DpDr from the standard cyclone dataframe output
                (saving space) and added a method for cyclone tracks to calculate the maximum distance
                from the genesis point that the cyclone is ever observed.
14 Jan 2021 --> Switch in when NaNs are counted, allowing minima over masked areas to interfere with nearby minima
            --> Removed an if statement from "findCenters" that I never actually use 
            (there is always an intensity threshold)
23 Mar 2021 --> Added rotateCoordsAroundOrigin function
15 Apr 2021 --> Modified the findCAP function to work as a post-hoc analysis
28 Apr 2021 --> Add linear regression function
05 May 2021 --> Remove "columns=" statements from initiation of empty pandas dataframes
29 May 2021 --> Fixed bug in linear regression function
05 Aug 2021 --> Added flexibility to circleKernel function so that the masked value can be NaN or 0
05 Oct 2021 --> Fixed bug in 360-day calendar calculations
'''
__version__ = "12.4"

import pandas as pd
import copy
import numpy as np
from scipy.spatial.distance import cdist
# from osgeo import gdal, gdalconst, gdalnumeric
from scipy import ndimage
from scipy import stats

'''
###############
### CLASSES ###
###############
'''

class minimum:
    '''This class stores the vital information about a minimum identified in a
    single field of data. It is used as a building block for more complicated 
    objets (cyclone, cyclonefield). Must provide the time, the x and y 
    locations of the minimum, and the value of the minimum (p).
    
    The type is used to identify the minimum as 0 = discarded from analysis; 
    1 = identified as a system center; 2 = identified as a secondary minimum 
    within a larger cyclone system.
    '''    
    def __init__(self,time,y,x,p_cent,i=0,t=0):
        self.time = time
        self.y = y
        self.x = x
        self.lat = np.nan
        self.long = np.nan
        self.p_cent = p_cent
        self.id = i
        self.tid = np.nan
        self.p_edge = p_cent
        self.area = 0
        self.areaID = np.nan
        self.DsqP = np.nan
        self.type = t
        self.secondary = [] # stores the ids of any secondary minima
        self.parent = {"y":y,"x":x,"id":i}
        self.precip = 0
        self.precipArea = 0
    def radius(self):
        return np.sqrt(self.area/(np.pi))
    def depth(self):
        return self.p_edge-self.p_cent
    def Dp(self):
        return self.depth()/self.radius()
    def centerCount(self):
        if self.type == 1:
            return len(self.secondary)+1
        else:
            return 0
    def add_parent(self,yPar,xPar,iPar):
        self.type = 2
        self.parent = {"y":yPar,"x":xPar,"id":iPar}

class cyclonefield:
    '''This class stores a) binary fields showing presence/absence of cyclone 
    centers and cyclone areas and b) an object for each cyclone -- all at a 
    particular instant in time. Good for computing summary values of the 
    cyclone state at a particular time. By default, the elevations parameters 
    are set to 0 (ignored in analysis).
    
    To initiate, must define:\n
    time = any format, but [Y,M,D,H,M,S] is suggested
    field = a numpy array of SLP (usu. in Pa; floats or ints)\n
    
    May also include:\n
    max_elev = the maximum elevation at which SLP minima will be considered (if
        you intend on changing the default, must also load a DEM array)
    elevField = an elevation array that must have the same shape as field\n
    '''
    ##################
    # Initialization #
    ##################
    
    def __init__(self, time):
        self.time = time
        self.cyclones = []
       
    #############################
    # Cyclone Centers Detection #
    #############################
    def findMinima(self, field, mask, kSize, nanthreshold=0.5):
        '''Identifies minima in the field using limits and kSize.
        '''
        self.fieldMinima = detectMinima(field,mask,kSize,nanthreshold).astype(np.int)
    def findCenters(self, field, mask, kSize, nanthreshold, d_slp, d_dist, yDist, xDist, lats, lons):
        '''Uses the initialization parameters to identify potential cyclones
        centers. Identification begins with finding minimum values in the field
        and then uses a gradient parameter and elevation parameter to restrict
        the number of centers. A few extra characteristics about the minima are 
        recorded (e.g. lat, long, Laplacian). Lastly, it adds a minimum object 
        to the centers list for each center identified in the centers field.
        '''
        # STEP 1: Calculate Laplacian
        laplac = laplacian(field)        
        
        # STEP 2: Identify Centers
        self.fieldCenters = findCenters(field, mask, kSize, nanthreshold, d_slp, d_dist, yDist, xDist)
        
        # Identify center locations
        rows, cols = np.where(self.fieldCenters > 0)
        
        # STEP 3: Assign each as a minimum in the centers list:
        for c in range(np.sum(self.fieldCenters)):
            center = minimum(self.time,rows[c],cols[c],field[rows[c],cols[c]],c,0)
            center.lat = lats[rows[c],cols[c]]
            center.long = lons[rows[c],cols[c]]
            center.DsqP = laplac[rows[c],cols[c]]
            self.cyclones.append(center)
            
    ###############################
    # Cyclone Area Identification #
    ###############################
    def findAreas(self, fieldMask, contint, mcctol, mccdist, lats, lons, kSize):
        # Identify maxima
        maxes = detectMaxima(fieldMask,fieldMask,kSize)
        
        # Define Areas, Identify Primary v. Secondary Cyclones
        self.fieldAreas, self.fieldCenters, self.cyclones = \
            findAreas(fieldMask, self.fieldCenters, self.cyclones,\
            contint, mcctol, mccdist, lats, lons, maxes)
        
        # Identify area id for each center
        cAreas, nC = ndimage.measurements.label(self.fieldAreas)
        for i in range(len(self.cyclones)):
            self.cyclones[i].areaID = cAreas[self.cyclones[i].y,self.cyclones[i].x]
    
    ######################
    # Summary Statistics #
    ######################
    # Summary values:
    def cycloneCount(self):
        return len([c.type for c in self.cyclones if c.type == 1])
    def area_total(self):
        areas = self.area()
        counts = self.centerCount()
        return sum([areas[a] for a in range(len(areas)) if counts[a] > 0])
    
    # Reorganization of cyclone object values:    
    def x(self):
        return [c.x for c in self.cyclones]
    def y(self):
        return [c.y for c in self.cyclones]
    def lats(self):
        return [c.lat for c in self.cyclones]
    def longs(self):
        return [c.long for c in self.cyclones]
    def p_cent(self):
        return [c.p_cent for c in self.cyclones]
    def p_edge(self):
        return [c.p_edge for c in self.cyclones]
    def radius(self):
        return [c.radius for c in self.cyclones]
    def area(self):
        return [c.area for c in self.cyclones]
    def areaID(self):
        return [c.areaID for c in self.cyclones]
    def centerType(self):
        return [c.type for c in self.cyclones]
    def centerCount(self):
        return [c.centerCount() for c in self.cyclones]
    def tid(self):
        return [c.tid for c in self.cyclones]

class cyclonetrack:
    '''This class stores vital information about the track of a single cyclone.
    It contains a pandas dataframe built from objets of the class minimum 
    and provides summary statistics about the cyclone's track and life cycle. 
    To make sense, the cyclone objects should be entered in chronological order.
    
    UNITS:
    time = days
    x, y, dx, dy = grid cells (e.g., 1 = 100 km)
    area = sq grid cells (e.g., 1 = 100 km^2)
    p_cent, p_edge, depth = Pa
    radius = grid cells
    u, v, uv = km/hr
    DsqP = Pa/(grid cell)^2
    DpDt = Pa/day
    id, tid, sid, ftid, otid, centers, type = no units
        id = a unique number for each center identified in a SLP field
        tid = a unique number for each center track in a given month
        ptid = the tid of the parent center in a MCC (ptid == tid is single-
            center cyclones)
        ftid = the tid of a center in the prior month (only applicable if a
            cyclone has genesis in a different month than its lysis)
        otid = the tid of a cyclone center that interacts with the given
            center (split, merge, re-genesis)
    ly, ge, rg, sp, mg = 0: no event occurred, 1: only a center-related event 
        occurred, 2: only an area-related event occurred, 3: both center- and 
        area-related events occurred
    '''
    ##############
    # Initialize #
    ##############
    def __init__(self,center,tid,Etype=3, ptid=np.nan, ftid=np.nan):
        self.tid = tid # A track id
        self.ftid = ftid # Former track id
        self.ptid = ptid # The most current parent track id
        
        # Create Main Data Frame
        self.data = pd.DataFrame()
        row0 = pd.DataFrame([{"time":center.time, "id":center.id, "pid":center.parent["id"],\
            "x":center.x, "y":center.y, "lat":center.lat, "long":center.long, \
            "p_cent":center.p_cent, "p_edge":center.p_edge, "area":center.area, \
            "radius":center.radius(), "depth":center.depth(),\
            "DsqP":center.DsqP,"type":center.type, "centers":center.centerCount(),\
            "Ege":Etype,"Erg":0,"Ely":0,"Esp":0,"Emg":0,"ptid":ptid},])
        self.data = self.data.append(row0, ignore_index=1, sort=1)
        
        # Create Events Data Frame
        self.events = pd.DataFrame()
        event0 = pd.DataFrame([{"time":center.time,"id":center.id,"event":"ge",\
            "Etype":Etype,"otid":np.nan,"x":center.x,"y":center.y},])
        self.events = self.events.append(event0, ignore_index=1, sort=1)
    
    ###############
    # Append Data #
    ###############   
    def addInstance(self,center,ptid=-1):
        Dt = center.time - self.data.time.iloc[-1] # Identify the time step
        
        if Dt != 0:
            Dp = center.p_cent - self.data.p_cent.iloc[-1]
            Dx = center.x - self.data.x.iloc[-1]
            Dy = center.y - self.data.y.iloc[-1]
            u = haversine(center.lat,center.lat,self.data.long.iloc[-1],center.long)/(Dt*1000*24)
            v = haversine(self.data.lat.iloc[-1],center.lat,center.long,center.long)/(Dt*1000*24)
            uv = haversine(self.data.lat.iloc[-1],center.lat,self.data.long.iloc[-1],center.long)/(Dt*1000*24)
            # Following Roebber (1984) and Serreze et al. (1997), scale the deepening rate by latitude
            DpDt = (Dp/Dt) * (np.sin(np.pi/3)/np.sin(np.pi*center.lat/180))
        
        if ptid == -1: # If no ptid is given, set it to the track's current ptid.
            ptid = self.ptid
        
        row = pd.DataFrame([{"time":center.time, "id":center.id, "pid":center.parent["id"],\
            "x":center.x, "y":center.y, "lat":center.lat, "long":center.long, \
            "p_cent":center.p_cent, "p_edge":center.p_edge, "area":center.area, \
            "radius":center.radius(), "depth":center.depth(),\
            "DsqP":center.DsqP,"type":center.type, "centers":center.centerCount(),\
            "Dp":Dp, "Dx":Dx, "Dy":Dy, "u":u, "v":v, "uv":uv, "DpDt":DpDt, "ptid":ptid,\
            "Ege":0,"Ely":0,"Esp":0,"Emg":0,"Erg":0},])
        
        self.data = self.data.append(row, ignore_index=1, sort=1)
    
    def removeInstance(self,time):
        '''Removes an instance from the main data frame and the events data 
        frame given a time. Note, this will remove mulitple events if they 
        occur at the same time. Time is in units of days.
        '''
        self.data = self.data.drop(self.data.index[self.data.time == time])
        self.events = self.events.drop(self.events.index[self.events.time == time])
    
    def addEvent(self,center,time,event,Etype,otid=np.nan):
        '''Events include genesis (ge), regenesis (rg) splitting (sp), merging
        (mg), and lysis (ly). Splitting and merging require the id of the 
        cyclone track being split from or merged with (otid). Note that lysis 
        is given the time step and location of the last instance of the 
        cyclone. For all types except rg, the event can be center-based, area-
        based, or both. Genesis occurs when a center/area doesn't exist in 
        time 1 but does exist in time 2. Lysis occurs when a center/area does 
        exist in time 1 but doesn't in time 2. A split occurs when one center/
        area in time 1 tracks to multiple centers/areas in time 2. A merge 
        occurs when multiple centers/areas in time 1 track to the same center/
        area in time 2. Regenesis is a special type of area genesis that occurs
        if the primary system of multiple centers experiences lysis but the
        system continues on from a secondary center.
        
        The occurrence of events is recorded both in an events data frame and
        the main tracking data frame.
        
        center = an object of class minimum that represents a cyclone center
        event = ge, rg, ly, sp, or mg
        eType = 1: center only, 2: area only, 3: both center and area
        otid = the track id of the other center involved for sp and mg events.
        '''
        row = pd.DataFrame([{"time":time,"id":center.id,"event":event,\
            "Etype":Etype,"otid":otid,"x":center.x,"y":center.y},])
        self.events = self.events.append(row, ignore_index=1, sort=1)
        
        # Event Booleans for Main Data Frame
        if event == "ge":
            self.data.loc[self.data.time == time,"Ege"] = Etype
        elif event == "ly":
            self.data.loc[self.data.time == time,"Ely"] = Etype
        elif event == "sp":
            self.data.loc[self.data.time == time,"Esp"] = Etype
        elif event == "mg":
            self.data.loc[self.data.time == time,"Emg"] = Etype
        elif event == "rg":
            self.data.loc[self.data.time == time,"Erg"] = Etype
    
    #############
    # Summarize #
    #############
    def lifespan(self):
        '''Subtracts the earliest time stamp from the latest.'''
        return np.max(list(self.data.time)) - np.min(list(self.data.loc[self.data.type != 0,"time"]))
    def maxDpDt(self):
        '''Returns the maximum deepening rate in the track and the 
        time and location (row, col) in which it occurred.'''
        v = np.min(np.where(np.isfinite(list(self.data.DpDt)) == 1,self.data.DpDt,np.inf))
        t = list(self.data.loc[self.data.DpDt == v,"time"])
        y = [int(i) for i in self.data.loc[self.data.DpDt == v,"y"]]
        x = [int(i) for i in self.data.loc[self.data.DpDt == v,"x"]]
        return v, t, y, x
    def maxDsqP(self):
        '''Returns the maximum intensity in the track and the time 
        and location (row, col) in which it occurred.'''
        v = np.max(np.where(np.isfinite(list(self.data.DsqP)) == 1,self.data.DsqP,-np.inf))
        t = list(self.data.loc[self.data.DsqP == v,"time"])
        y = [int(i) for i in self.data.loc[self.data.DsqP == v,"y"]]
        x = [int(i) for i in self.data.loc[self.data.DsqP == v,"x"]]
        return v, t, y, x
    def minP(self):
        '''Returns the minimum pressure in the track and the 
        time and location (row, col) in which it occurred.'''
        v = np.min(np.where(np.isfinite(list(self.data.p_cent)) == 1,self.data.p_cent,np.inf))
        t = list(self.data.loc[self.data.p_cent == v,"time"])
        y = [int(i) for i in self.data.loc[self.data.p_cent == v,"y"]]
        x = [int(i) for i in self.data.loc[self.data.p_cent == v,"x"]]
        return v, t, y, x
    def maxUV(self):
        '''Returns the maximum cyclone propagation speed in the track and the 
        time and location (row, col) in which it occurred.'''
        v = np.max(np.where(np.isfinite(list(self.data.uv)) == 1,self.data.uv,-np.inf))
        t = list(self.data.loc[self.data.uv == v,"time"])
        y = [int(i) for i in self.data.loc[self.data.uv == v,"y"]]
        x = [int(i) for i in self.data.loc[self.data.uv == v,"x"]]
        return v, t, y, x
    def maxDepth(self):
        '''Returns the maximum depth in the track and the 
        time and location (row, col) in which it occurred.'''
        v = np.max(np.where(np.isfinite(list(self.data.depth)) == 1,self.data.depth,-np.inf))
        t = list(self.data.loc[self.data.depth == v,"time"])
        y = [int(i) for i in self.data.loc[self.data.depth == v,"y"]]
        x = [int(i) for i in self.data.loc[self.data.depth == v,"x"]]
        return v, t, y, x
    def trackLength(self):
        '''Adds together the distance between each segment of the track to find
        the total distance traveled (in kms).'''
        t = 24*(self.data.time.iloc[1] - self.data.time.iloc[0]) # Hours between timestep
        return t*self.data.loc[((self.data.type != 0) | (self.data.Ely > 0)),"uv"].sum()
    def maxDistFromGenPnt(self):
        '''Returns the maximum distance a cyclone is ever observed from its
        genesis point in units of km.'''
        v = np.max([haversine(self.data.lat[0],self.data.lat[i],self.data.long[0],self.data.long[i]) for i in range(len(self.data.long))])
        return v/1000
    def avgArea(self):
        '''Identifies the average area for the track and the time stamp for 
        when it occurred.'''
        areas = [float(i) for i in self.data.loc[self.data.type != 0,"area"]]
        return float(sum(areas))/len(self.data.loc[self.data.type != 0,"area"])
    def mcc(self):
        '''Returns a 1 if at any point along the track the cyclone system is
        a multi-center cyclone. Retruns a 0 otherwise.'''
        if np.nansum([int(c) != 1 for c in self.data.centers.loc[self.data.type != 0]]) == 0:
            return 0
        else:
            return 1
    def CAP(self):
        '''Returns the total cyclone-associated precipitation for the cyclone center.'''
        return np.nansum( list(self.data.loc[self.data.type != 0,"precip"]) )

class systemtrack:
    '''This class stores vital information about the track of a single system.
    It contains a pandas dataframe built from objets of the class minimum 
    and provides summary statistics about the system's track and life cycle. 
    To make sense, the system track should be constructed directly from 
    finished cyclone tracks. The difference between a system track and a 
    cyclone track is that a cyclone track exists for each cyclone center, 
    whereas only one system track exists for each mcc.
    
    UNITS:
    time = days
    x, y, dx, dy = grid cells (1 = 100 km)
    area, precipArea = sq grid cells (1 = (100 km)^2)
    p_cent, p_edge, depth = Pa
    radius = grid cells
    u, v, uv = km/hr
    DsqP = Pa/(grid cell)^2
    DpDt = Pa/day
    id, tid, sid, ftid, otid, centers, type = no units
        id = a unique number for each center identified in a SLP field
        tid = a unique number for each center track in a given month
        sid = a unique number for each system track in a given month
        ptid = the tid of the parent center in a MCC (ptid == tid is single-
            center cyclones)
        ftid = the tid of a center in the prior month (only applicable if a
            cyclone has genesis in a different month than its lysis)
        otid = the tid of a cyclone center that interacts with the given
            center (split, merge, re-genesis)
    ly, ge, rg, sp, mg = 0: no event occurred, 1: only a center-related event 
        occurred, 2: only an area-related event occurred, 3: both center- and 
        area-related events occurred
    '''
    ##############
    # Initialize #
    ##############
    def __init__(self,data,events,tid,sid,ftid=np.nan):
        self.tid = tid # A track id
        self.ftid = ftid # The former track id
        self.sid = sid # A system id
                
        # Create Main Data Frame
        self.data = copy.deepcopy(data)
        
        # Create Events Data Frame
        self.events = copy.deepcopy(events)
    
    #############
    # Summarize #
    #############
    def lifespan(self):
        '''Subtracts the earliest time stamp from the latest.'''
        return np.max(list(self.data.time)) - np.min(list(self.data.loc[self.data.type != 0,"time"]))
    def maxDpDt(self):
        '''Returns the maximum deepening rate in the track and the 
        time and location (row, col) in which it occurred.'''
        v = np.min(np.where(np.isfinite(list(self.data.DpDt)) == 1,self.data.DpDt,np.inf))
        t = list(self.data.loc[self.data.DpDt == v,"time"])
        y = [int(i) for i in self.data.loc[self.data.DpDt == v,"y"]]
        x = [int(i) for i in self.data.loc[self.data.DpDt == v,"x"]]
        return v, t, y, x
    def maxDsqP(self):
        '''Returns the maximum intensity in the track and the time 
        and location (row, col) in which it occurred.'''
        v = np.max(np.where(np.isfinite(list(self.data.DsqP)) == 1,self.data.DsqP,-np.inf))
        t = list(self.data.loc[self.data.DsqP == v,"time"])
        y = [int(i) for i in self.data.loc[self.data.DsqP == v,"y"]]
        x = [int(i) for i in self.data.loc[self.data.DsqP == v,"x"]]
        return v, t, y, x
    def minP(self):
        '''Returns the minimum pressure in the track and the 
        time and location (row, col) in which it occurred.'''
        v = np.min(np.where(np.isfinite(list(self.data.p_cent)) == 1,self.data.p_cent,np.inf))
        t = list(self.data.loc[self.data.p_cent == v,"time"])
        y = [int(i) for i in self.data.loc[self.data.p_cent == v,"y"]]
        x = [int(i) for i in self.data.loc[self.data.p_cent == v,"x"]]
        return v, t, y, x
    def maxUV(self):
        '''Returns the maximum cyclone propagation speed in the track and the 
        time and location (row, col) in which it occurred.'''
        v = np.max(np.where(np.isfinite(list(self.data.uv)) == 1,self.data.uv,-np.inf))
        t = list(self.data.loc[self.data.uv == v,"time"])
        y = [int(i) for i in self.data.loc[self.data.uv == v,"y"]]
        x = [int(i) for i in self.data.loc[self.data.uv == v,"x"]]
        return v, t, y, x
    def maxDepth(self):
        '''Returns the maximum depth in the track and the 
        time and location (row, col) in which it occurred.'''
        v = np.max(np.where(np.isfinite(list(self.data.depth)) == 1,self.data.depth,-np.inf))
        t = list(self.data.loc[self.data.depth == v,"time"])
        y = [int(i) for i in self.data.loc[self.data.depth == v,"y"]]
        x = [int(i) for i in self.data.loc[self.data.depth == v,"x"]]
        return v, t, y, x
    def trackLength(self):
        '''Adds together the distance between each segment of the track to find
        the total distance traveled (in kms).'''
        t = 24*(self.data.time.iloc[1] - self.data.time.iloc[0]) # Hours between timestep
        return t*self.data.loc[((self.data.type != 0) | (self.data.Ely > 0)),"uv"].sum()
    def maxDistFromGenPnt(self):
        '''Returns the maximum distance a cyclone is ever observed from its
        genesis point in units of km.'''
        v = np.max([haversine(self.data.lat[0],self.data.lat[i],self.data.long[0],self.data.long[i]) for i in range(len(self.data.long))])
        return v/1000
    def avgArea(self):
        '''Identifies the average area for the track and the time stamp for 
        when it occurred.'''
        areas = [float(i) for i in self.data.loc[self.data.type != 0,"area"]]
        return float(sum(areas))/len(self.data.loc[self.data.type != 0,"area"])
    def mcc(self):
        '''Returns a 1 if at any point along the track the cyclone system is
        a multi-center cyclone. Retruns a 0 otherwise.'''
        if np.nansum([int(c) != 1 for c in self.data.centers.loc[self.data.type != 0]]) == 0:
            return 0
        else:
            return 1
    def CAP(self):
        '''Returns the total cyclone-associated precipitation for the cyclone center.'''
        return np.nansum( list(self.data.loc[self.data.type != 0,"precip"]) )

'''        
#################
### FUNCTIONS ###
#################
'''

'''###########################
Find Nearest Value
###########################'''
def findNearest(array,value):
    '''
    Finds the gridcell of a numpy array that most closely matches the given
    value. Returns value and its index.
    '''
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx

'''###########################
Find Nearest Point
###########################'''
def findNearestPoint(a,B,latlon=0):
    '''
    Finds the closest location in the array B to the point a, when a is an 
    ordered pair (row,col or lat,lon) and B is an array or list of ordered 
    pairs. All pairs should be in the same format (row,col) or (col,row).
    The optional paramter latlon is 0 by default, meaning that "closest" is
    calculated as a Euclidian distance of numpy array positions. If latlon=1, 
    the haversine formula is used to determine distance instead, which means 
    all ordered pairs should be (lat,lon).
    
    Returns the index of the location in B that is closest and the minimum distance.
    '''
    if latlon == 0:
        dist = [( (b[0]-a[0])**2 + (b[1]-a[1])**2 )**0.5 for b in B]
    else:
        dist = [haversine(a[0],b[0],a[1],b[1]) for b in B]
    i = np.argmin(dist)
    
    return i , dist[i]

'''###########################
Find Nearest Area
###########################'''
def findNearestArea(a,B,b="all",latlon=[]):
    '''
    Finds the closest unique area in the array B to the point a, when a is an 
    ordered pair (row,col or lat,lon) and B is an array of contiguous areas 
    identified by a unique integer.* All pairs should be in the same format 
    (row,col). 
    
    The optional parameter b can be used to assess a subset of the areas in B 
    (in which case b should be a list, tuple, or 1-D arrray). By default, all 
    areas in B are assessed.
    
    The optional parameter latlon is [] by default, meaning that "closest" is
    calculated as a Euclidian distance of numpy array positions. Alternatively,
    latlon can be a list of two numpy arrays (lats,lons) with the same shape 
    as the input B. If so, the haversine formula is used to determine distance
    instead. If latitude and longitude are used, then a should be a tuple of
    latitude and longitude; otherwise, it should be a tuple of (row,col)
    
    Returns the ID of the area in B that is closest.
    
    *You can generate an array like this from a field of 0s and 1s using 
    scipy.ndimage.measurements.label(ARRAY).
    '''
    if b == "all":
        b = np.unique(B)[np.where(np.unique(B) != 0)]
    
    # First identify the shortest distance between point a and each area in B
    if latlon == []:
        dist = []
        for j in b:
            locs1 = np.where(B == j)
            locs2 = [(locs1[0][i],locs1[1][i]) for i in range(len(locs1[0]))]
            dist.append(findNearestPoint(a,locs2)[1])
    else:
        dist = []
        for j in b:
            locs1 = np.where(B == j)
            locs2 = [(latlon[0][locs1[0][i],locs1[1][i]],latlon[1][locs1[0][i],locs1[1][i]]) for i in range(len(locs1[0]))]
            dist.append(findNearestPoint(a,locs2,1)[1])
        
    # Then identify the shortest shortest distance
    return b[np.argmin(dist)]

'''###########################
Leap Year Boolean Creation
###########################'''
def leapyearBoolean(years):
    '''
    Given a list of years, this function will identify which years are leap 
    years and which years are not. Returns a list of 0s (not a leap year) and 
    1s (leap year). Each member of the year list must be an integer or float.
    
    Requires numpy.
    '''
    ly = [] # Create empty list
    for y in years: # For each year...
        if (y%4 == 0) and (y%100!= 0): # If divisible by 4 but not 100...
            ly.append(1) # ...it's a leap year
        elif y%400 == 0: # If divisible by 400...
            ly.append(1) # ...it's a leap year
        else: # Otherwise...
            ly.append(0) # ...it's NOT a leap year
    
    return ly

'''###########################
Calculate Days Between Two Dates
###########################'''
def daysBetweenDates(date1,date2,lys=1,dpy=365,nmons=12):
    '''
    Calculates the number of days between date1 (inclusive) and date2 (exclusive)
    when given dates in list format [year,month,day,hour,minute,second] or 
    [year,month,day]. Works even if one year is BC (BCE) and the other is AD (CE). 
    If hours are used, they must be 0 to 24. Requires numpy.
    
    date1 = the start date (earlier in time; entire day included in the count if time of day not specified)\n
    date2 = the end date (later in time; none of day included in count unless time of day is specified)
    
    lys = 0 for no leap years, 1 for leap years (+1 day in Feb) as in the Gregorian calendar
    dpy = days per year in a non-leap year (defaults to 365)
    nmons = number of months per year; if not using the Gregorian calendar, this must be a factor of dpy
    '''
    if dpy == 365:
        db4 = [0,31,59,90,120,151,181,212,243,273,304,334] # Number of days in the prior months
    else:
        db4 = list(np.arange(nmons)*int(dpy/nmons))
    
    if date1[0] == date2[0]: # If the years are the same...
        # 1) No intervening years, so ignore the year value:        
        daysY = 0
    
    else: # B) If the years are different...
        
        # 1) Calculate the total number of days based on the years given:
        years = range(date1[0],date2[0]) # make a list of all years to count
        years = [yr for yr in years if yr != 0]
        if lys==1:
            lyb = leapyearBoolean(years) # annual boolean for leap year or not leap year
        else:
            lyb = [0]
        
        daysY = dpy*len(years)+np.sum(lyb) # calculate number of days
    
    if lys == 1:
        ly1 = leapyearBoolean([date1[0]])[0]
        ly2 = leapyearBoolean([date2[0]])[0]
    else:
        ly1, ly2 = 0, 0
    
    # 2) Calcuate the total number of days to subtract from start year
    days1 = db4[date1[1]-1] + date1[2] -1 # days in prior months + prior days in current month - the day you're starting on
    # Add leap day if appropriate:    
    if date1[1] > 2:
        days1 = days1 + ly1
    
    # 3) Calculate the total number of days to add from end year
    days2 = db4[date2[1]-1] + date2[2] - 1 # days in prior months + prior days in current month - the day you're ending on
    # Add leap day if appropriate:    
    if date2[1] > 2:
        days2 = days2 + ly2
        
    # 4) Calculate fractional days (hours, minutes, seconds)
    day1frac, day2frac = 0, 0
    
    if len(date1) == 6:
        day1frac = (date1[5] + date1[4]*60 + date1[3]*3600)/86400.
    elif len(date1) != 3:
        raise Exception("date1 does not have the correct number of values.")
    
    if len(date2) == 6:
        day2frac = (date2[5] + date2[4]*60 + date2[3]*3600)/86400.
    elif len(date2) != 3:
        raise Exception("date2 does not have the correct number of values.")
    
    # 5) Final calculation
    days = daysY - days1 + days2 - day1frac + day2frac
    
    return days

'''###########################
Add Time
###########################'''
def timeAdd(time1,time2,lys=1,dpy=365):
    '''This function takes the sum of two times in the format [Y,M,D,H,M,S]. 
    The variable time1 should be a proper date (Months are 1 to 12, Hours are 0 to 23), 
    but time2 does not have to be a proper date. Note that if you use months or years in time2, 
    the algorithm will not discrimnate the number of days per month or year. To be precise,
    use only days, hours, minutes, and seconds. It can handle the BC/AD transition but 
    can only handle non-integers for days, hours, minutes, and seconds. To
    perform date subtraction, simply make the entries in time2 negative numbers.
    
    lys = Boolean to determine whether to recognize leap years (1; default) 
    dpy = Days per yera (non leap years) -- must be 365 (default) or some multiple of 30 (e.g., 360).
    --> Note: If there are leap years (lys = 1), dpy is forced to be 365
    
    Addition Examples:
    [2012,10,31,17,44,54] + [0,0,0,6,15,30] = [2012,11,1,0,0,24] #basic example
    [2012,10,31,17,44,54] + [0,0,0,0,0,22530] = [2012,11,1,0,0,24] #time2 is improper time
    
    [1989,2,25,0,0,0] + [0,5,0,0,0,0] = [1989,7,25,0,0,0] #non-leap year months
    [1988,2,25,0,0,0] + [0,5,0,0,0,0] = [1988,7,25,0,0,0] #leap year months
    
    [1989,2,25,0,0,0] + [0,0,150,0,0,0] = [1989,7,25,0,0,0] #non-leap year days
    [1988,2,25,0,0,0] + [0,0,150,0,0,0] = [1988,7,24,0,0,0] #leap year days
    
    [1989,7,25,0,0,0] + [4,0,0,0,0] = [1993,7,25,0,0,0] #non-leap year years
    [1988,7,25,0,0,0] + [4,0,0,0,0] = [1992,7,25,0,0,0] #leap year years
    
    [-1,12,31,23,59,59] + [0,0,0,0,0,1] = [1,1,1,0,0,0] #crossing BC/AD with seconds
    [-2,1,1,0,0,0] + [4,0,0,0,0,0] = [3,1,1,0,0,0] #crossing BC/AD with years
    
    [1900,9,30,12,0,0] + [0,0,0.25,0,0,0] = [1900, 9, 30.0, 18, 0, 0.0] #fractional days
    [1900,9,30,12,0,0] + [0,0,0.5,0,0,0] = [1900, 10, 1.0, 0, 0, 0.0] #fractional days
    
    Subtraction Examples:
    [2012,10,31,17,44,54] - [0,0,0,-10,-50,-15] = [2012,10,31,6,54,39] #basic example
    [2012,10,31,17,44,54] - [0,0,0,0,0,-39015] = [2012,10,31,6,54,39] #time2 is imporper time
    
    [1989,7,25,0,0,0] + [0,-5,0,0,0,0] = [1989,2,25,0,0,0] #non-leap year months
    [1988,7,25,0,0,0] + [0,-5,0,0,0,0] = [1988,2,25,0,0,0] #leap year months
    
    [1989,7,25,0,0,0] + [0,0,-150,0,0,0] = [1989,2,25,0,0,0] #non-leap year days
    [1988,7,25,0,0,0] + [0,0,-150,0,0,0] = [1988,2,26,0,0,0] #leap year days
    
    [1993,2,25,0,0,0] + [-4,0,0,0,0] = [1989,2,25,0,0,0] #non-leap year years
    [1992,2,25,0,0,0] + [-4,0,0,0,0] = [1988,2,25,0,0,0] #leap year years
    
    [1,1,1,0,0,0] + [0,0,0,0,0,-1] = [-1,12,31,23,59,59] #crossing BC/AD with seconds
    [2,1,1,0,0,0] + [-4,0,0,0,0,0] = [-3,1,1,0,0,0] #crossing BC/AD with years 
    
    [1900,9,30,12,0,0] + [0,0,0.25,0,0,0] = [1900, 9, 30.0, 6, 0, 0.0] #fractional days
    [1900,9,30,12,0,0] + [0,0,0.5,0,0,0] = [1900, 9, 29.0, 0, 0, 0.0] #fractional days
    '''
    if len(time1) == 3:
        time1 = time1 + [0,0,0]
    if len(time2) == 3:
        time2 = time2 + [0,0,0]
    
    # Ensure that years and months are whole numbers:
    if time1[0]%1 != 0 or time1[1]%1 != 0 or time2[0]%1 != 0 or time2[1]%1 != 0:
        raise ValueError("The year and month entries are not all whole numbers.")
    
    else:
        # Identify Fractional days:
        day1F = time1[2]%1
        day2F = time2[2]%1
        
        # Initial Calculation: Add, transfer appropriate amount to next place, keep the remainder
        secR = (time1[5] + time2[5] + (day1F+day2F)*86400)%60
        minC = np.floor((time1[5] + time2[5] + (day1F+day2F)*86400)/60)
        
        minR = (time1[4] + time2[4] + minC)%60
        hrsC = np.floor((time1[4] + time2[4] + minC)/60)
        
        hrsR = (time1[3] + time2[3] + hrsC)%24
        dayC = np.floor((time1[3] + time2[3] + hrsC)/24)
        
        dayA = (time1[2]-day1F) + (time2[2]-day2F) + dayC # Initially, just calculate days
        
        monA = (time1[1]-1 + time2[1])%12 + 1 # Because there is no month 0
        yrsC = np.floor((time1[1]-1 + time2[1])/12)
        
        yrsA = time1[0] + time2[0] + yrsC
        
        ######################
        #### REFINEMENTS  ####
        dpm = [31,28,31,30,31,30,31,31,30,31,30,31] # days per month
        dpmA = [d for d in dpm] # make modifiable copy
        
        ### Gregorian Calendar ###
        if lys == 1:
            ### Deal with BC/AD ###
            if time1[0] < 0 and yrsA >= 0: # Going from BC to AD
                yrsR = yrsA + 1
            elif time1[0] > 0 and yrsA <= 0: # Going from AD to BC
                yrsR = yrsA - 1
            else:
                yrsR = yrsA
            
            ### Deal with Days ###
            dpmA[1] = dpmA[1] + leapyearBoolean([yrsR])[0] # days per month adjusted for leap year (if applicable)
            
            if dayA > 0: # if the number of days is positive
                if dayA <= dpmA[monA-1]: # if the number of days is positive and less than the full month...
                    dayR = dayA #...no more work needed
                    monR = monA
                
                elif dayA <= sum(dpmA[monA-1:]): # if the number of days is positive and over a full month but not enough to carry over to the next year...
                    monR = monA
                    dayR = dayA
                    while dayR > dpmA[monR-1]: # then walk through each month, subtracting days as you go until there's less than a month's worth
                        dayR = dayR - dpmA[monR-1]
                        monR = monR+1
                
                else: # if the number of days is positive and will carry over to another year...
                    dayR = dayA - sum(dpmA[monA-1:]) # go to Jan 1 of next year...
                    yrsR = yrsR+1
                    ly = leapyearBoolean([yrsR])[0]
                    while dayR > 365+ly: # and keep subtracting 365 or 366 (leap year dependent) until until no longer possible
                        dayR = dayR - (365+ly)
                        yrsR = yrsR+1
                        if yrsR == 0: # Disallow 0-years
                            yrsR = 1
                        ly = leapyearBoolean([yrsR])[0]
                    
                    dpmB = [d for d in dpm]
                    dpmB[1] = dpmB[1] + ly
                    monR = 1
                    while dayR > dpmB[monR-1]: # then walk through each month
                        dayR = dayR - dpmB[monR-1]
                        monR = monR+1
            
            elif dayA == 0: # if the number of days is 0
                if monA > 1:
                    monR = monA-1
                    dayR = dpmA[monR-1]
                else:
                    monR = 12
                    dayR = 31
                    yrsR = yrsR - 1
            
            else: # if the number of days is negative
                if abs(dayA) < sum(dpmA[:monA-1]): # if the number of days will stay within the same year...
                    monR = monA
                    dayR = dayA
                    while dayR <= 0:
                        monR = monR-1
                        dayR = dpmA[monR-1] + dayR
                
                else: # if the number of days is negative and will cross to prior year...
                    dayR = dayA + sum(dpmA[:monA-1])
                    yrsR = yrsR-1
                    if yrsR == 0:
                        yrsR = -1
                    
                    ly = leapyearBoolean([yrsR])[0]
                    while abs(dayR) >= 365+ly:
                        dayR = dayR + (365+ly)
                        yrsR = yrsR-1
                        ly = leapyearBoolean([yrsR])[0]
                    
                    dpmB = [d for d in dpm]
                    dpmB[1] = dpmB[1] + ly
                    monR = 13
                    dayR = dayR
                    while dayR <= 0:
                        monR = monR-1
                        dayR = dpmB[monR-1] + dayR
        
        ### 365-Day Calendar ###
        elif dpy == 365:            
            if dayA > 0: # if the number of days is positive
                if dayA <= dpmA[monA-1]: # if the number of days is positive and less than the full month...
                    dayR = dayA #...no more work needed
                    monR = monA
                
                elif dayA <= sum(dpmA[monA-1:]): # if the number of days is positive and over a full month but not enough to carry over to the next year...
                    monR = monA
                    dayR = dayA
                    while dayR > dpmA[monR-1]: # then walk through each month, subtracting days as you go until there's less than a month's worth
                        dayR = dayR - dpmA[monR-1]
                        monR = monR+1
                
                else: # if the number of days is positive and will carry over to another year...
                    dayR = dayA - sum(dpmA[monA-1:]) # go to Jan 1 of next year...
                    yrsA = yrsA+1
                    while dayR > 365: # and keep subtracting 365 until until no longer possible
                        dayR = dayR - 365
                        yrsA = yrsA+1
                    
                    dpmB = [d for d in dpm]
                    dpmB[1] = dpmB[1]
                    monR = 1
                    while dayR > dpmB[monR-1]: # then walk through each month
                        dayR = dayR - dpmB[monR-1]
                        monR = monR+1
            
            elif dayA == 0: # if the number of days is 0
                if monA > 1:
                    monR = monA-1
                    dayR = dpmA[monR-1]
                else:
                    monR = 12
                    dayR = 31
                    yrsA = yrsA -1
            
            else: # if the number of days is negative
                if abs(dayA) < sum(dpmA[:monA-1]): # if the number of days will stay within the same year...
                    monR = monA
                    dayR = dayA
                    while dayR <= 0:
                        monR = monR-1
                        dayR = dpmA[monR-1] + dayR
                
                else: # if the number of days is negative and will cross to prior year...
                    dayR = dayA + sum(dpmA[:monA-1])
                    yrsA = yrsA-1
                    while abs(dayR) >= 365:
                        dayR = dayR + 365
                        yrsA = yrsA-1
                    
                    dpmB = [d for d in dpm]
                    dpmB[1] = dpmB[1]
                    monR = 13
                    dayR = dayR
                    while dayR <= 0:
                        monR = monR-1
                        dayR = dpmB[monR-1] + dayR
            
            ### Deal with BC/AD ###
            if time1[0] < 0 and yrsA >= 0: # Going from BC to AD
                yrsR = yrsA + 1
            elif time1[0] > 0 and yrsA <= 0: # Going from AD to BC
                yrsR = yrsA - 1
            else:
                yrsR = yrsA
        
        ### 360-Day Calendar ### (or other mulitple of 30) ###
        else:            
            if dayA > 0: # if the number of days is positive
                if dayA <= 30: # if the number of days is positive and less than the full month...
                    monR = monA
                    dayR = dayA #...no more work needed
                
                elif (dayA + (monA-1)*30) <= dpy: # if the number of days is positive and over a full month but not enough to carry over to the next year...
                    monR = monA + int(dayA/30) # add months
                    dayR = dayA%30 # find new day-of-month
                
                else: # if the number of days is positive and will carry over to another year...
                    yrsA = yrsA+1
                    dayR = (monA-1)*30 + dayA - dpy # go to Jan 1 of next year...
                    
                    yrsA = yrsA + int(dayR/dpy) # add years
                    dayR = dayR%dpy # find new day-of-year

                    monR = int(dayR/30) + 1 # add months
                    dayR = dayR%30 # find new day-of-month
                    
            elif dayA == 0: # if the number of days is 0
                if monA > 1:
                    monR = monA-1
                else:
                    monR = int(dpy/30)
                    yrsA = yrsA -1
                dayR = 30
            
            else: # if the number of days is negative
                if abs(dayA) < (monA-1)*30: # if the number of days will stay within the same year...
                    monR = monA-1 + int(dayA/30) # Subtract months
                    dayR = dayA%30 # Find new number of days
                
                else: # if the number of days is negative and will cross to prior year...
                    yrsA = yrsA-1
                    dayR = dayA + (monA-1)*30 # go to Dec 30 of prior year
                    
                    # find new day of year
                    yrsA = yrsA + int(dayR/dpy) # subtract years
                    dayR = dayR%dpy # find new day of year (switches to positive)
                    
                    monR = int(dayR/30) + 1  # add months
                    dayR = dayR%30 # find new day of month
            
            if dayR == 0:
                if monR == 1:
                    yrsA = yrsA-1
                    monR = int(dpy/30)
                else:
                    monR = monR-1
                dayR = 30
            
            ### Deal with BC/AD ###
            if time1[0] < 0 and yrsA >= 0: # Going from BC to AD
                yrsR = yrsA + 1
            elif time1[0] > 0 and yrsA <= 0: # Going from AD to BC
                yrsR = yrsA - 1
            else:
                yrsR = yrsA
        
        return [int(yrsR),int(monR),int(dayR),int(hrsR),int(minR),secR]
    
'''###########################
Calculate a Latitudinal Angle along a Certain Distance of a Meridian
###########################'''
def dist2lat(d,units="km", r=6371.):
    '''This function converts from standard distance units to a latitudinal
    angle on a sphere when given its radius. By default, the distance is assumed
    to be in kilometers and the radius is assumed to be 6371 km (i.e., the 
    sphere is Earth). Returns the latitudinal angle in degrees. Note: this
    function only works if working strictly meridional.
    '''
    import numpy as np
    
    # Other Conversions:
    km = ["km","kms","kilometer","kilometre","kilometers","kilometres"]
    m = ["m","ms","meter","meters","metres","metres"]
    ft = ["ft","feet"]
    mi = ["mi","miles","mile","mi"]
    nm = ["nm","nautical mile","nms","nautical miles","mile nautical",\
        "miles nautical","mile (nautical)","miles (nautical)"]
    
    if units.lower() in km:
        d = d
    elif units.lower() in ft:
        d = d/3280.84
    elif units.lower() in mi:
        d = d/0.621371
    elif units.lower() in nm:
        d = d/0.5399568
    elif units.lower() in m:
        d = d/1000
    
    # Main calculation
    return (180/np.pi)*(d/r)

'''###########################
Calculate a Longitudinal Angle along a Certain Distance of a Parallel
###########################'''
def dist2long(d, lat1, lat2, units="km", r=6371.):
    '''This function converts from standard distance units to a longitudinal 
    angle  on a sphere when given two latitudes (in degrees) and the radius of 
    the sphere using the haversine formula. By default, the distance is assumed
    to be in kilometers and the radius is assumed to be 6371 km (i.e., the 
    sphere is Earth). Returns the longitudinal angle in degrees.
    '''
    import numpy as np
    
    # Convert latitudes to radians:
    lat1, lat2 = lat1*np.pi/180, lat2*np.pi/180
    
    # Other Conversions:
    km = ["km","kms","kilometer","kilometre","kilometers","kilometres"]
    m = ["m","ms","meter","meters","metres","metres"]
    ft = ["ft","feet"]
    mi = ["mi","miles","mile","mi"]
    nm = ["nm","nautical mile","nms","nautical miles","mile nautical",\
        "miles nautical","mile (nautical)","miles (nautical)"]
    
    if units.lower() in km:
        d = d
    elif units.lower() in ft:
        d = d/3280.84
    elif units.lower() in mi:
        d = d/0.621371
    elif units.lower() in nm:
        d = d/0.5399568
    elif units.lower() in m:
        d = d/1000
    
    # Main calculation
    dlat = lat2 - lat1
    c = d/r
    dlon = 2*np.arcsin(( ( np.sin(c/2)**2 - np.sin(dlat/2)**2 ) / ( np.cos(lat1)*np.cos(lat2) ) )**0.5)*180.0/np.pi
    #dlon = 2*np.arcsin( np.sin(c/2) / np.cos(lat1) ) # for constant latitude
    
    return dlon

'''###########################
Add Degrees of Latitude
###########################'''   
def addLat(lat0,dlat):
    '''Adds a change in latitude to an initial latitude; can handle crossing
    the poles.\n
    lat0 = initial latitude (degrees)
    dlat = change in latitude (degrees)
    '''
    lat1 = lat0 + dlat
    if lat1 > 90:
        lat1 = 180 - lat1
    elif lat1 < -90:
        lat1 = -180 - lat1
    return lat1

'''###########################
Add Degrees of Longitude
###########################'''  
def addLong(long0,dlong,neglong=1):
    '''Adds a change in longitude to an initial longitude; can handle crossing
    the -180/180 or 0/360 boundary.\n
    long0 = initial longitude
    dlong = change in longitude
    neglong = binary; 1 means longitude goes from -180 to 180 (default);
    0 means longitude goes from 0 to 360 (i.e. no negative longitudes)
    '''
    long1 = long0 + dlong
    
    if neglong == 0:
        if long1 > 360:
            long1 = long1 - 360
        elif long1 < 0:
            long1 = long1 + 360
    else:
        if long1 > 180:
            long1 = long1 - 360
        elif long1 < -180:
            long1 = long1 + 360
    
    return long1

'''###########################
Rotate Coordinates Around the Origin
###########################'''
def rotateCoordsAroundOrigin(x1,y1,phi):
    '''This function will identify the new coordinates (x2,y2) of a point with 
    coordinates (x1,y1) rotated by phi (radians) around the origin (0,0). The 
    outputs are the two new coordinates x2 and y2.  Note that if using numpy arrays, 
    "y" corresponds to the row position and "x" to the column position.
    
    x1, y1 = the x and y coordinates of the original point (integers or floats)\n
    phi = the desired rotation from -pi to +pi radians
    '''
    # Calculate the distance from the origin:
    c = np.sqrt(np.square(y1) + np.square(x1))
    # Rotate X coordinate:
    x2 = c*np.cos(np.arctan2(y1,x1)-phi)
    # Rotate Y coordinate:
    y2 = c*np.sin(np.arctan2(y1,x1)-phi)
    
    return x2,y2

'''###########################
Ring Kernel Creation
###########################'''
def ringKernel(ri,ro,d=0):
    '''Given two radii in numpy array cells, this function will calculate a 
    numpy array of 1 and nans where 1 is the cells whose centroids are more than 
    ri units away from the center centroid but no more than ro units away. The 
    result is a field of nans with a ring of 1s.
    
    ri = inner radius in numpy array cells (integer or float)
    ro = outer radius in numpy array cells (integer or float)
    d = if d==0, then function returns an array of 1s and nans
        if d==1, then function returns an array of distances from
        center (in array cells)
    
    '''
    # Create a numpy array of 1s:
    k = int(ro*2+1)
    ringMask=np.ones((k,k))
    
    # If returing 1s and nans:
    if d == 0:
        for row in range(0,k):
            for col in range(0,k):
                d = ((row-ro)**2 + (col-ro)**2)**0.5
                if d > ro or d <= ri:
                    ringMask[row,col]=np.NaN
        return ringMask
    
    # If returning distances:
    if d == 1:
        ringDist = np.zeros((k,k))
        for row in range(0,k):
            for col in range(0,k):
                ringDist[row,col] = ((row-ro)**2 + (col-ro)**2)**0.5
                if ringDist[row,col] > ro or ringDist[row,col] <= ri:
                    ringMask[row,col]=np.NaN
        ringDist = ringMask*ringDist
        return ringDist

def ringDistance(ydist, xdist, rad):
    '''Given the grid cell size in the x and y direction and a desired radius, 
    creates a numpy array for which the distance from the center is recorded
    in all cells between a distance of rad and rad-mean(ydist,xdist) and all
    other cells are np.nan.
    
    ydist, xdist = the grid cell size in the x and y direction
    rad = the desired radius -- must be same units as xdist and ydist
    '''
    # number of cells in either cardinal direction of the center of the kernel
    kradius = np.int( np.ceil( rad / min(ydist,xdist) ) )
    # number of cells in each row and column of the kernel
    kdiameter = kradius*2+1

    # for each cell, calculate x distances and y distances from the center
    kxdists = np.tile(np.arange(-kradius, kradius + 1), (kdiameter, 1)) * xdist
    kydists = np.rot90(np.tile(np.arange(-kradius, kradius + 1), (kdiameter, 1)) * ydist)

    # apply pythagorean theorem to calculate euclidean distances
    kernel = np.sqrt(np.square(kxdists) + np.square(kydists))

    # create a boolean mask which determines the cells that should be nan
    mask = (kernel > rad) | (kernel <= (rad - np.mean((ydist, xdist))))

    kernel[mask] = np.nan

    return kernel

'''###########################
Circle Kernel Creation
###########################'''
def circleKernel(r,masked_value=np.nan):
    '''Given the radius in numpy array cells, this function will calculate a 
    numpy array of 1 and some other value where 1 is the cells whose centroids 
    are less than the radius away from the center centroid.
    
    r = radius in numpy array cells (integer or float)
    masked_value = value to use for cells whose centroids are more than r
    distnace away from the center centroid. np.nan by default
    '''
    # Create a numpy array of 1s:
    rc = int(np.ceil(r))
    k = rc*2+1
    circleMask=np.ones((k,k))
    for row in range(0,k):
        for col in range(0,k):
            d = ((row-rc)**2 + (col-rc)**2)**0.5
            if d > r:
                circleMask[row,col]=masked_value
    return circleMask

'''###########################
Calculate a Smoothed Density Field
###########################'''
def smoothField(var,kSize,nanedge=0):
    '''Uses a rectangular kernel to smooth the input numpy array.  Dimensions
    of the kernel are set by kSize.
    
    var = a numpy array of values to be smoothed
    kSize = a tuple or list of the half-height and half-width of the kernel or,
        if the kernel is a square, this may be a single number representing half-width
    nanedge = binary that indicates treatment of edges; if 0, nans are ignored in calculations, 
        if 1, the smoothed value will be a nan if any nans are observed in the kernel
        
    # WARNING: Deprecated starting in Version 12 of code. Use scipy.ndimage.uniform_filter instead.
    '''
    
    # Create kernel
    if (isinstance(kSize,tuple) == 0) and (isinstance(kSize,list) == 0):
        krow = range(-kSize,kSize+1)
        kcol = range(-kSize,kSize+1)
    else:
        krow = range(-kSize[0],kSize[0]+1)
        kcol = range(-kSize[1],kSize[1]+1)
    
    # Store main dimensions
    nrow = var.shape[0]
    ncol = var.shape[1]
    
    # Create list of arrays for smoothing
    varFields = []
    for r in krow:
        for c in kcol:
            varFields.append( np.hstack(( np.zeros((nrow+kSize*2, kSize-c))*np.nan , \
            np.vstack(( np.zeros((kSize-r,ncol))*np.nan, var ,np.zeros((kSize+r,ncol))*np.nan )), \
            np.zeros((nrow+kSize*2, kSize+c))*np.nan )) )
    
    # Smooth arrays
    if nanedge == 1:
        smoothedField = np.apply_along_axis(np.mean,0,varFields,dtype="float")
    else:
        smoothedField = np.apply_along_axis(np.nanmean,0,varFields,dtype="float")
    
    # Return smooothed array in original dimensions
    return smoothedField[kSize:nrow+kSize,kSize:ncol+kSize]

def filterField(var,kSize,nanedge=0,style="mean"):
    '''Uses a rectangular kernel to filter/smooth the input numpy array.  
    Dimensions of the kernel are set by kSize.
    
    var = a numpy array of values to be smoothed
    kSize = a tuple or list of the half-height and half-width of the kernel or,
        if the kernel is a square, this may be a single number repsresenting half-width
    nanedge = binary that indicates treatment of edges; if 0, nans are ignored in calculations, 
        if 1, the smoothed value will be a nan if any nans are observed in the kernel
    style = the style of filter, can be mean (or average), median, maximum (or max), 
    minimum (or min), sum (or total), or standard deviation (std). Note that there is currently no
    "ignore nan" option for standard deviation
    '''
    # Create kernel
    if (isinstance(kSize,tuple) == 0) and (isinstance(kSize,list) == 0): # Squares
        krow = range(-kSize,kSize+1)
        kcol = range(-kSize,kSize+1)
    else: # Rectangles
        krow = range(-kSize[0],kSize[0]+1)
        kcol = range(-kSize[1],kSize[1]+1)
    
    # Store main dimensions
    nrow = var.shape[0]
    ncol = var.shape[1]
    
    # Create list of arrays for smoothing
    varFields = []
    for r in krow:
        for c in kcol:
            varFields.append( np.hstack(( np.zeros((nrow+kSize*2, kSize-c))*np.nan , \
            np.vstack(( np.zeros((kSize-r,ncol))*np.nan, var ,np.zeros((kSize+r,ncol))*np.nan )), \
            np.zeros((nrow+kSize*2, kSize+c))*np.nan )) )
    
    # Smooth arrays
    if nanedge == 1:
        if (style.lower() == "mean") or (style.lower() == "average"):
            smoothedField = np.apply_along_axis(np.mean,0,varFields,dtype="float")
        elif (style.lower() == "sum") or (style.lower() == "total"):
            smoothedField = np.apply_along_axis(np.sum,0,varFields,dtype="float")
        elif (style.lower() == "median"):
            smoothedField = np.apply_along_axis(np.median,0,varFields,dtype="float")
        elif (style.lower() == "max") or (style.lower() == "maximum"):
            smoothedField = np.apply_along_axis(np.max,0,varFields,dtype="float")
        elif (style.lower() == "min") or (style.lower() == "minimum"):
            smoothedField = np.apply_along_axis(np.min,0,varFields,dtype="float")
        elif style.lower() in ["sd","std","stdev","stddev","standard deviation"]:
            smoothedField = np.apply_along_axis(np.std,0,varFields,dtype="float")
        else:
            raise Exception("Invalid style passed to filterField function. Valid styles include mean, average, median, minimum, min, maximum, max, sum, total, standard deviation, sd, std, stdec, and stddev")
    else:
        if (style.lower() == "mean") or (style.lower() == "average"):
            smoothedField = np.apply_along_axis(np.nanmean,0,varFields,dtype="float")
        elif (style.lower() == "sum") or (style.lower() == "total"):
            smoothedField = np.apply_along_axis(np.nansum,0,varFields,dtype="float")
        elif (style.lower() == "median"):
            smoothedField = np.apply_along_axis(np.nanmedian,0,varFields,dtype="float")
        elif (style.lower() == "max") or (style.lower() == "maximum"):
            smoothedField = np.apply_along_axis(np.nanmax,0,varFields,dtype="float")
        elif (style.lower() == "min") or (style.lower() == "minimum"):
            smoothedField = np.apply_along_axis(np.nanmin,0,varFields,dtype="float")
        elif style.lower() in ["sd","std","stdev","stddev","standard deviation"]:
            smoothedField = np.apply_along_axis(np.std,0,varFields,dtype="float")
            raise Warning("No 'ignore nans' option for standard deviation; nans propagated.")
        else:
            raise Exception("Invalid style passed to filterField function. Valid styles include mean, average, median, minimum, min, maximum, max, sum, total, standard deviation, sd, std, stdec, and stddev")
    
    # Return smooothed array in original dimensions
    return smoothedField[kSize:nrow+kSize,kSize:ncol+kSize]

'''###########################
Calculate Mean Gradient from Center Based on a Kernel
###########################'''
def kernelGradient(field,location,yDist,xDist,rad):
    '''Given a field and a tuple or list of (row,col), this function
    calculates the difference between the location and each gridcell in a
    subset that matches the kernel. It then calculates the gradient using 
    the x and y distances and calculates the mean for all gridcells in the 
    kernel.
    
    field = a numpy array of floats or integers
    location = a tuple or list in the format (row,col) in array coordinates
    yDist = a numpy array of distances between rows
    xDist = a numpy array of distance between columns
    rad =  radius in real units (must match units of yDist and xDist)
    
    Returns a mean gradient in field units per radius units
    '''
    # Define row & col
    row, col = location[0], location[1]
    
    # Define ring distance kernel & kernel size
    kernel = ringDistance(yDist[row,col],xDist[row,col],rad)
    ksize = int((kernel.shape[0]-1)/2)
    
    # Define the starting and stopping rows and cols:
    rowS = row-ksize
    rowE = row+ksize
    colS = col-ksize
    colE = col+ksize
    if rowS < 0:
        rowS = 0
    if rowE > field.shape[0]:
        rowE = field.shape[0]
    if colS < 0:
        colS = 0
    if colE > field.shape[1]:
        colE = field.shape[1]
    
    # Take a subset to match the kernel size
    subset = field[rowS:rowE+1,colS:colE+1]
    
    # Add nans to fill out subset if not already of equal size with kernel
    if kernel.shape[0] > subset.shape[0]: # If there aren't enough rows to match kernel
        nanrows = np.empty( ( (kernel.shape[0]-subset.shape[0]),subset.shape[1] ) )*np.nan
        if rowS == 0: # Add on top if the first row is row 0
            subset = np.vstack( (nanrows,subset) )
        else: # Add to bottom otherwise
            subset = np.vstack( (subset,nanrows) )
    if kernel.shape[1] > subset.shape[1]: # If there aren't enough columns to match kernel
        nancols = np.empty( (subset.shape[0], (kernel.shape[1]-subset.shape[1]) ) )*np.nan
        if colS == 0: # Add to left if first col is col 0
            subset = np.hstack( (nancols,subset) )
        else: # Add to right otherwise
            subset = np.hstack( (subset,nancols) )
    
    # Apply kernel and calculate difference between each cell and the center
    subset_ring = (subset - field[row,col])/kernel
    
    # Find the mean value (excluding nans)
    mean = np.nanmean(subset_ring)
    
    return mean

'''###########################
Calculate Sobel Gradients of Field
###########################'''
def slope_latlong(var, lat, lon, edgeLR = "calc", edgeTB = "calc", dist="km"):
    '''This function returns numpy arrays of the meridional and zonal
    variable gradient and its magnitude when fed numpy arrays for
    the variable, latitude, and longitude at each point.  A sobel operator is 
    used to calculate slope.  Cells do not have to be square because cell size
    is calculated from the latitude and longitude rasters, but cell size must 
    be constant. Output units are the units of var over the units of dist.\n  
    
    var = variable field (numpy array of integers or floats); such as temperature
    or humidity\n
    lats = latitude field (numpy array of latitudes)\n
    longs = longitude field (numpy array of longitudes)\n
    dist = the units of the denominator in the output (meters by default,
        also accepts kilometers, miles, or nautical miles)\n
    edgeLR, edgeTB = behavior of the gradient calculation at the left and right
        hand edges and top and bottom edges of the numpy array, respectively.
        Choose "calc" to claculate a partial Sobel operator, "wrap" if opposite
        edges are adjacent, "nan" to set the edges to a nan value, and "zero"
        to set the edges to a zero value. By default, the function will calculate 
        a gradient based on a partial Sobel operator. Use "wrap" if the dataset 
        spans -180 to 180 longitude. Use "zeros" if the dataset spans -90 to 
        90 latitude (i.e. distance is zero at the edge).
    '''
    #################
    # Set up Arrays #
    #################
    # Find shape of input field
    nrow = var.shape[0]
    ncol = var.shape[1]
    
    if edgeLR == "wrap" and edgeTB == "wrap":
        var1 = np.vstack(( np.hstack((var[-2:,-2:], var[-2:,:])), np.hstack((var[:,-2:] , var)) ))
        var2 = np.vstack(( np.hstack((var[-2:,-1:], var[-2:,:], var[-2:,0:1])), np.hstack((var[:,-1:] , var, var[:,0:1])) ))
        var3 = np.vstack(( np.hstack((var[-2:,:], var[-1:,0:2])), np.hstack((var, var[:,0:2])) ))
        var4 = np.vstack(( np.hstack((var[-1:,-2:], var[-1:,:])), np.hstack((var[:,-2:] , var)), np.hstack((var[0:1,-2:], var[0:1,:])) ))
        var5 = np.vstack(( np.hstack((var[-1:,-1:], var[-1:,:], var[-1:,0:1])), np.hstack((var[:,-1:] , var, var[:,0:1])), np.hstack((var[0:1,-1:], var[0:1,:], var[0:1,0:1])) ))
        var6 = np.vstack(( np.hstack((var[-1:,:], var[-1:,0:2])), np.hstack((var, var[:,0:2])), np.hstack((var[0:1,:], var[0:1,0:2])) ))
        var7 = np.vstack(( np.hstack((var[:,-2:] , var)), np.hstack((var[0:2,-2:], var[0:2,:])) ))
        var8 = np.vstack(( np.hstack((var[:,-1:] , var, var[:,0:1])), np.hstack((var[0:2,-1:], var[0:2,:], var[0:2,0:1])) ))
        var9 = np.vstack(( np.hstack((var, var[:,0:2])), np.hstack((var[0:2,:], var[0:2,0:2])) ))
        
        lon4 = np.vstack(( np.hstack((lon[-1:,-2:], lon[-1:,:])), np.hstack((lon[:,-2:] , lon)), np.hstack((lon[0:1,-2:], lon[0:1,:])) ))
        lon6 = np.vstack(( np.hstack((lon[-1:,:], lon[-1:,0:2])), np.hstack((lon, lon[:,0:2])), np.hstack((lon[0:1,:], lon[0:1,0:2])) ))
        lon2 = np.vstack(( np.hstack((lon[-2:,-1:], lon[-2:,:], lon[-2:,0:1])), np.hstack((lon[:,-1:] , lon, lon[:,0:1])) ))
        lon8 = np.vstack(( np.hstack((lon[:,-1:] , lon, lon[:,0:1])), np.hstack((lon[0:2,-1:], lon[0:2,:], lon[0:2,0:1])) ))
        
        lat4 = np.vstack(( np.hstack((lat[-1:,-2:], lat[-1:,:])), np.hstack((lat[:,-2:] , lat)), np.hstack((lat[0:1,-2:], lat[0:1,:])) ))
        lat6 = np.vstack(( np.hstack((lat[-1:,:], lat[-1:,0:2])), np.hstack((lat, lat[:,0:2])), np.hstack((lat[0:1,:], lat[0:1,0:2])) ))        
        lat2 = np.vstack(( np.hstack((lat[-2:,-1:], lat[-2:,:], lat[-2:,0:1])), np.hstack((lat[:,-1:] , lat, lat[:,0:1])) ))
        lat8 = np.vstack(( np.hstack((lat[:,-1:] , lat, lat[:,0:1])), np.hstack((lat[0:2,-1:], lat[0:2,:], lat[0:2,0:1])) ))
    
    elif edgeLR == "wrap" and edgeTB != "wrap":
        var1 = np.vstack(( np.zeros((2,ncol+2)), np.hstack((var[:,-2:] , var)) ))
        var2 = np.vstack(( np.zeros((2,ncol+2)), np.hstack((var[:,-1:] , var, var[:,0:1])) ))
        var3 = np.vstack(( np.zeros((2,ncol+2)), np.hstack((var, var[:,0:2])) ))
        var4 = np.vstack(( np.zeros((1,ncol+2)), np.hstack((var[:,-2:] , var)), np.zeros((1,ncol+2)) ))
        var5 = np.vstack(( np.zeros((1,ncol+2)), np.hstack((var[:,-1:] , var, var[:,0:1])), np.zeros((1,ncol+2)) ))
        var6 = np.vstack(( np.zeros((1,ncol+2)), np.hstack((var, var[:,0:2])), np.zeros((1,ncol+2)) ))
        var7 = np.vstack(( np.hstack((var[:,-2:] , var)), np.zeros((2,ncol+2)) ))
        var8 = np.vstack(( np.hstack((var[:,-1:] , var, var[:,0:1])), np.zeros((2,ncol+2)) ))
        var9 = np.vstack(( np.hstack((var, var[:,0:2])), np.zeros((2,ncol+2)) ))
        
        lon4 = np.vstack(( np.zeros((1,ncol+2)), np.hstack((lon[:,-2:] , lon)), np.zeros((1,ncol+2)) ))
        lon6 = np.vstack(( np.zeros((1,ncol+2)), np.hstack((lon, lon[:,0:2])), np.zeros((1,ncol+2)) ))
        lon2 = np.vstack(( np.zeros((2,ncol+2)), np.hstack((lon[:,-1:] , lon, lon[:,0:1])) ))
        lon8 = np.vstack(( np.hstack((lon[:,-1:] , lon, lon[:,0:1])), np.zeros((2,ncol+2)) ))
        
        lat4 = np.vstack(( np.zeros((1,ncol+2)), np.hstack((lat[:,-2:] , lat)), np.zeros((1,ncol+2)) ))
        lat6 = np.vstack(( np.zeros((1,ncol+2)), np.hstack((lat, lat[:,0:2])), np.zeros((1,ncol+2)) ))        
        lat2 = np.vstack(( np.zeros((2,ncol+2)), np.hstack((lat[:,-1:] , lat, lat[:,0:1])) ))
        lat8 = np.vstack(( np.hstack((lat[:,-1:] , lat, lat[:,0:1])), np.zeros((2,ncol+2)) ))
    
    elif edgeLR != "wrap" and edgeTB == "wrap":
        var1 = np.hstack(( np.zeros((nrow+2,2)) , np.vstack((var[-2:,:],var)) ))
        var2 = np.hstack(( np.zeros((nrow+2,1)) , np.vstack((var[-2:,:],var)) , np.zeros((nrow+2,1)) ))
        var3 = np.hstack(( np.vstack((var[-2:,:],var)) , np.zeros((nrow+2,2)) ))
        var4 = np.hstack(( np.zeros((nrow+2,2)) , np.vstack((var[-1:,:],var,var[0:1,:])) ))
        var5 = np.hstack(( np.zeros((nrow+2,1)) , np.vstack((var[-1:,:],var,var[0:1,:])) , np.zeros((nrow+2,1)) ))
        var6 = np.hstack(( np.vstack((var[-1:,:],var,var[0:1,:])), np.zeros((nrow+2,2)) ))
        var7 = np.hstack(( np.zeros((nrow+2,2)) , np.vstack((var,var[0:2,:])) ))
        var8 = np.hstack(( np.zeros((nrow+2,1)) , np.vstack((var,var[0:2,:])) , np.zeros((nrow+2,1)) ))
        var9 = np.hstack(( np.vstack((var,var[0:2,:])) , np.zeros((nrow+2,2)) ))
        
        lat4 = np.hstack(( np.zeros((nrow+2,2)) , np.vstack((lat[-1:,:],lat,lat[0:1,:])) ))
        lat6 = np.hstack(( np.vstack((lat[-1:,:],lat,lat[0:1,:])), np.zeros((nrow+2,2)) ))        
        lat2 = np.hstack(( np.zeros((nrow+2,1)) , np.vstack((lat[-2:,:],lat)) , np.zeros((nrow+2,1)) ))
        lat8 = np.hstack(( np.zeros((nrow+2,1)) , np.vstack((lat,lat[0:2,:])) , np.zeros((nrow+2,1)) ))
        
        lon4 = np.hstack(( np.zeros((nrow+2,2)) , np.vstack((lon[-1:,:],lon,lon[0:1,:])) ))
        lon6 = np.hstack(( np.vstack((lon[-1:,:],lon,lon[0:1,:])), np.zeros((nrow+2,2)) ))
        lon2 = np.hstack(( np.zeros((nrow+2,1)) , np.vstack((lon[-2:,:],lon)) , np.zeros((nrow+2,1)) ))
        lon8 = np.hstack(( np.zeros((nrow+2,1)) , np.vstack((lon,lon[0:2,:])) , np.zeros((nrow+2,1)) ))
    
    else:
        var1 = np.hstack(( np.zeros((nrow+2,2)) , np.vstack((np.zeros((2,ncol)),var)) ))
        var2 = np.hstack(( np.zeros((nrow+2,1)) , np.vstack((np.zeros((2,ncol)),var)) , np.zeros((nrow+2,1)) ))
        var3 = np.hstack(( np.vstack((np.zeros((2,ncol)),var)) , np.zeros((nrow+2,2)) ))
        var4 = np.hstack(( np.zeros((nrow+2,2)) , np.vstack((np.zeros((1,ncol)),var,np.zeros((1,ncol)))) ))
        var5 = np.hstack(( np.zeros((nrow+2,1)) , np.vstack((np.zeros((1,ncol)),var,np.zeros((1,ncol)))) , np.zeros((nrow+2,1)) ))
        var6 = np.hstack(( np.vstack((np.zeros((1,ncol)),var,np.zeros((1,ncol)))), np.zeros((nrow+2,2)) ))
        var7 = np.hstack(( np.zeros((nrow+2,2)) , np.vstack((var,np.zeros((2,ncol)))) ))
        var8 = np.hstack(( np.zeros((nrow+2,1)) , np.vstack((var,np.zeros((2,ncol)))) , np.zeros((nrow+2,1)) ))
        var9 = np.hstack(( np.vstack((var,np.zeros((2,ncol)))) , np.zeros((nrow+2,2)) ))
        
        lat2 = np.hstack(( np.zeros((nrow+2,1)) , np.vstack((np.zeros((2,ncol)),lat)) , np.zeros((nrow+2,1)) ))
        lat4 = np.hstack(( np.zeros((nrow+2,2)) , np.vstack((np.zeros((1,ncol)),lat,np.zeros((1,ncol)))) ))
        lat6 = np.hstack(( np.vstack((np.zeros((1,ncol)),lat,np.zeros((1,ncol)))), np.zeros((nrow+2,2)) ))
        lat8 = np.hstack(( np.zeros((nrow+2,1)) , np.vstack((lat,np.zeros((2,ncol)))) , np.zeros((nrow+2,1)) ))
        
        lon2 = np.hstack(( np.zeros((nrow+2,1)) , np.vstack((np.zeros((2,ncol)),lon)) , np.zeros((nrow+2,1)) ))        
        lon4 = np.hstack(( np.zeros((nrow+2,2)) , np.vstack((np.zeros((1,ncol)),lon,np.zeros((1,ncol)))) ))
        lon6 = np.hstack(( np.vstack((np.zeros((1,ncol)),lon,np.zeros((1,ncol)))), np.zeros((nrow+2,2)) ))
        lon8 = np.hstack(( np.zeros((nrow+2,1)) , np.vstack((lon,np.zeros((2,ncol)))) , np.zeros((nrow+2,1)) ))
    
    ##############################
    # Calculate Slope Components #
    ##############################
    # Find Cell Distances
    dX = haversine(lat4,lat6,lon4,lon6,dist)
    dY = haversine(lat8,lat2,lon8,lon2,dist)
    
    # Perform a Sobel Operator    
    dZdX = ( ((var3 + 2.*var6 + var9) - (var1 + 2.*var4 + var7)) / (4.*dX) )[1:nrow+1,1:ncol+1]
    dZdY = ( ((var1 + 2.*var2 + var3) - (var7 + 2.*var8 + var9)) / (4.*dY) )[1:nrow+1,1:ncol+1]
    
    # Edit TB Edges
    if edgeTB == "calc":
        dZdX[-1:,:] = ( ((var3[-2:-1,:] + 2.*var6[-2:-1,:]) - (var1[-2:-1,:] + 2.*var4[-2:-1,:])) / (3.*dX[-2:-1,:]) )[:,1:ncol+1]
        dZdX[0:1,:] = ( ((2.*var6[1:2,:] + var9[1:2,:]) - (2.*var4[1:2,:] + var7[1:2,:])) / (3.*dX[1:2,:]) )[:,1:ncol+1]
        dZdY[-1:,:] = ( ((var1[-2:-1,:] + 2.*var2[-2:-1,:] + var3[-2:-1,:]) - (var4[-2:-1,:] + 2.*var5[-2:-1,:] + var6[-2:-1,:])) / (2.*dY[-2:-1,:]) )[:,1:ncol+1]
        dZdY[0:1,:] = ( ((var4[-2:-1,:] + 2.*var5[-2:-1,:] + var6[-2:-1,:]) - (var7[-2:-1,:] + 2.*var8[-2:-1,:] + var9[-2:-1,:])) / (4.*dY[-2:-1,:]) )[:,1:ncol+1]
    elif edgeTB == "zero":
        dZdX[-1:,:], dZdX[0:1,:] = 0, 0
        dZdY[-1:,:], dZdY[0:1,:] = 0, 0
    elif edgeTB == "nan":
        dZdX[-1:,:], dZdX[0:1,:] = np.nan, np.nan
        dZdY[-1:,:], dZdY[0:1,:] = np.nan, np.nan
    
    # Edit LR Edges
    if edgeLR == "calc":
        dZdX[:,-1:] = ( ((var2[:,-2:-1] + 2.*var5[:,-2:-1] + var8[:,-2:-1]) - (var1[:,-2:-1] + 2.*var4[:,-2:-1] + var7[:,-2:-1])) / (2.*dX[:,-2:-1]) )[1:nrow+1,:]
        dZdX[:,0:1] = ( ((var3[:,1:2] + 2.*var6[:,1:2] + var9[:,1:2]) - (var2[:,1:2] + 2.*var5[:,1:2] + var8[:,1:2])) / (2.*dX[:,1:2]) )[1:nrow+1,:]
        dZdY[:,-1:] = ( ((var1[:,-2:-1] + 2.*var2[:,-2:-1]) - (var7[:,-2:-1] + 2.*var8[:,-2:-1])) / (3.*dY[:,-2:-1]) )[1:nrow+1,:]
        dZdY[:,0:1] = ( ((2.*var2[:,1:2] + var3[:,1:2]) - (2.*var8[:,1:2] + var9[:,1:2])) / (3.*dY[:,1:2]) )[1:nrow+1,:]
    elif edgeLR == "zero":
        dZdX[:,-1:], dZdX[:,0:1] = 0, 0
        dZdY[:,-1:], dZdY[:,0:1] = 0, 0    
    elif edgeTB == "nan":
        dZdX[-1:,:], dZdX[0:1,:] = np.nan, np.nan
        dZdY[-1:,:], dZdY[0:1,:] = np.nan, np.nan
    
    # Clean-up for cells that have lat of +/- 90
    cleaner = np.where(np.abs(lat) > 89.95,0,1)
    dZdX, dZdY = dZdX*cleaner, dZdY*cleaner # such cells are set to a gradient of 0!
    
    # Slope Magnitude
    slope = np.sqrt( np.square(dZdX) + np.square(dZdY) )  
    
    return slope, dZdX, dZdY

'''###########################
Calculate Laplacian of Field
###########################'''
def laplacian(field,multiplier=1):
    '''Given a field, this function calculates the Laplacian (similar to the field's 
    second derivative over two dimensions). The gradient is calculated using a 
    Sobel operator, so edge effects do exist. Following the method of Murray 
    and Simmonds (1991), the second derivative is first calculated for the x 
    and y orthognoal components individally and then combined to giv a divergence.
    
    field = a numpy array of values over which to take the Laplacian\n
    multiplier (optional) = an optional multiplier for converting units at the
    end of the calculation.  For example to convert from Pa/[100 km]^2 to 
    hPa/[100 km]^2, use 0.01.  The default is 1.
    '''
    # Calculate gradient with sobel operator
    sobY = ndimage.sobel(field,axis=0,mode='constant',cval=np.nan)/-8
    sobX = ndimage.sobel(field,axis=1,mode='constant',cval=np.nan)/8
    
    # Calcualte gradient of the gradient with sobel again 
    lapY = ndimage.sobel(sobY,axis=0,mode='constant',cval=np.nan)/-8
    lapX = ndimage.sobel(sobX,axis=1,mode='constant',cval=np.nan)/8
    
    # Add components
    laplac = (lapY + lapX)*multiplier

    return laplac

def laplacian_latlong(field,lats,lons,edgeLR = "calc", edgeTB = "calc", dist="km"):
    '''Given a field, this function calculates the Laplacian (the field's 
    second derivative over two dimensions). The gradient is calculated using a 
    Sobel operator, so edge effects do exist. Following the method of Murray 
    and Simmonds (1991), the second derivative is first calculated for the x 
    and y orthogonal components individally and then combined.
    
    field = a numpy array of values over which to take the Laplacian\n
    lats = a numpy array of latitudes (shape should match field)\n
    longs = a numpy array of longitudes (shape should match field)\n
    dist = the units of the denominator in the output (meters by default,
        also accepts kilometers, miles, or nautical miles)\n
    edgeLR, edgeTB = behavior of the gradient calculation at the left and right
        hand edges and top and bottom edges of the nupy array, respectively.
        Choose "calc" to claculate a partial Sobel operator, "wrap" if opposite
        edges are adjacent, "nan" to set the edges to a nan value, and "zero"
        to set the edges to a zero value. By default, the function will calculate 
        a gradient based on a partial Sobel operator. Use "wrap" if the dataset 
        spans -180 to 180 longitude. Use "zeros" if the dataset spans -90 to 
        90 latitude (i.e. distance is zero at the edge).
    '''
    # First Gradient
    slopeX, slopeY = slope_latlong(field, lats, lons, edgeLR, edgeTB, dist)[1:]
    
    # Second Gradient
    laplacX = slope_latlong(slopeX, lats, lons, edgeLR, edgeTB, dist)[1]
    laplacY = slope_latlong(slopeY, lats, lons, edgeLR, edgeTB, dist)[2]
    
    return laplacX + laplacY

'''###########################
Create Array Neighborhood
###########################'''
def arrayNeighborhood(var,kSize=3,edgeLR=0,edgeTB=0):
    '''
    '''
    k = kSize-1
    nrow = var.shape[0]
    ncol = var.shape[1]
    
    varcube = []
    
    if edgeLR != "wrap" and edgeTB != "wrap":
        for i in range(kSize):
            for j in range(kSize):
                varcube.append( np.hstack(( np.zeros((nrow+k,j))+edgeLR , np.vstack(( np.zeros((k-i,ncol))+edgeTB, var, np.zeros((i,ncol))+edgeTB )) , np.zeros((nrow+k,k-j))+edgeLR )) )

    elif edgeLR == "wrap" and edgeTB == "wrap":
         for i in range(kSize):
            for j in range(kSize):
                varcube.append( np.vstack(( np.hstack((var[-(k-i):,-(k-j):] , var[-(k-i):,:], var[-(k-i):,0:j])), np.hstack((var[:,-(k-j):], var, var[:,0:j])) , np.hstack((var[0:i,-(k-j):], var[0:i,:], var[0:i,0:j])) ))[:nrow+k,:ncol+k] )

    elif edgeLR == "wrap" and edgeTB != "wrap":
         for i in range(kSize):
            for j in range(kSize):
                varcube.append( np.vstack(( np.zeros((i,ncol+k))+edgeTB , np.hstack((var[:,-(k-j):], var, var[:,0:j]))[:,:ncol+k], np.zeros(((k-i),ncol+k))+edgeTB )) )
    
    else:
         for i in range(kSize):
            for j in range(kSize):
                varcube.append( np.hstack(( np.zeros((nrow+k,j))+edgeLR , np.vstack((var[-(k-i):,:], var, var[0:i,:]))[:nrow+k,:] , np.zeros((nrow+k,(k-j)))+edgeLR )) )

    return np.array(varcube)

'''###########################
Detect Minima and Maxima
###########################'''
def detectMinima(var, mask, kSize=3, nanthreshold=1):
    '''Identifies local minima in a numpy array (surface) by searching within a 
    square kernel (the neighborhood) for each grid cell. Ignores nans by default.
    
    field = a numpy array that represents some surface of interest
    mask = a numpy array of 0s and NaNs used to mask results -- can repeat var 
        input here if no mask exists
    kSize = kernel size (e.g., 3 means a 3*3 kernel centered on each grid cell)
    nanthreshold = maximum ratio of nan cells to total cells in the kernel for
        each minimum test. 0 means that no cell with a nan neighbor can be 
        considered a minimum. 0.5 means that less than half of the cells in the
        kernel can be nans for a cell to be considered a minimum. 1 means 
        that a cell will be considered a minimum even if all cells around it 
        are nan. Warning: since the center cell of the kernel is included in 
        the ratio, if you want to make decisions based on the ratio of nerighbors
        that exceed some threshold, scale your desired threshold as such: 
            nanthreshold = desiredthreshold * (kSize*kSize-1)/(kSize*kSize).
    '''
    # Find percentage of NaNs in each neighborhood
    nancount = ndimage.uniform_filter(np.isnan(mask).astype(np.float),kSize)

    # Find the local minima
    output = ndimage.minimum_filter(var,kSize)
    
    # export valid locations as 1s and invalid locations as 0s
    return (output == var) & (nancount < nanthreshold) & np.isfinite(mask)

def detectMaxima(var, mask, kSize=3, nanthreshold=1):
    '''Identifies local maxima in a numpy array (surface) by searching within a 
    square kernel (the neighborhood) for each grid cell. Ignores nans by default.
    
    field = a numpy array that represents some surface of interest
    mask = a numpy array of 0s and NaNs used to mask results -- can repeat var 
        input here if no mask exists
    kSize = kernel size (e.g., 3 means a 3*3 kernel centered on each grid cell)
    nanthreshold = maximum ratio of nan cells to total cells in the kernel for
        each maximum test. 0 means that no cell with a nan neighbor can be 
        considered a maximum. 0.5 means that less than half of the cells in the
        kernel can be nans for a cell to be considered a maximum. 1 means 
        that a cell will be considered a maximum even if all cells around it 
        are nan. Warning: since the center cell of the kernel is included in 
        the ratio, if you want to make decisions based on the ratio of nerighbors
        that exceed some threshold, scale your desired threshold as such: 
            nanthreshold = desiredthreshold * (kSize*kSize-1)/(kSize*kSize).
    '''
    # Find percentage of NaNs in each neighborhood
    nancount = ndimage.uniform_filter(np.isnan(mask).astype(np.float),kSize)

    # Find the local minima
    output = ndimage.maximum_filter(var,kSize)
    
    # export valid locations as 1s and invalid locations as 0s
    return (output == var) & (nancount < nanthreshold) & np.isfinite(mask)

'''###########################
Find Special Types of Minima of a Surface
###########################'''
def findCenters(field, mask, kSize=3, nanthreshold=0.5, d_slp=0, d_dist=100, yDist=0, xDist=0):
    '''This function identifies minima in a field and then eliminates minima
    that do not satisfy a gradient parameter. By default, the function will 
    ONLY find minima.
    
    field = the numpy array that you want to find the minima of.\n
    kSize = the kernel size that should be considered
        when determining whether it is a minimum; 3 by default\n
    nanthreshold = maximum ratio of nan cells to total cells around the center
        cell for each minimum test. 0 means that no cell with a nan neighbor 
        can be considered a minimum. 0.5 means that less than half of the 
        neighbors can be nans for a cell to be considered a minimum. 1 means 
        that a cell will be considered a minimum if all cells around it are nan.\n    
    d_slp and d_dist = the SLP and distance that determine the minimum pressure
        gradient allowed for a minimum to be considered a cyclone center. By 
        default they are left at 0, so no gradients will be considered.\n
    yDist, xDist = numpy arrays of distances between rows and columns. Only 
        necessary if calculating gradients, so the default is 0.\n
    '''
    # STEP 1.1. Identify Minima:
    fieldMinima = detectMinima(field,mask,kSize,nanthreshold=nanthreshold).astype(np.uint8)
    
    # STEP 1.2. Discard Weak Minima:        
    rowMin, colMin = np.where(fieldMinima == 1) # Identify locations of minima
    sysMin = fieldMinima.copy() # make a mutable copy of the minima locations
    
    d_grad = float(d_slp)/d_dist # Define gradient limit
    for sm in range(len(rowMin)): # For each minimum...
        # Calculate gradient:
        mean_gradient = kernelGradient(field+mask,(rowMin[sm],colMin[sm]),yDist,xDist,d_dist)
#            print(str(sm) +": mean_gradient: " + str(mean_gradient) + ",  d_grad: " + str(d_grad))
        # Test for pressure gradient:
        if (mean_gradient < d_grad):
            sysMin[rowMin[sm],colMin[sm]] = 0
    
        # Save centers
        fieldCenters = sysMin
    
    return fieldCenters

'''###########################
Identify the Areas Associated with Special Minima of a Surface
###########################'''
def findAreas(field,fieldCenters,centers,contint,mcctol,mccdist,lats,lons,maxes=0):
    '''This function actually does more than define the areas of influence for 
    individual cyclone centers -- it identifies the areas of influence for
    entire storm systems, which means it includes the detection of multi-center
    cyclones(MCCs). The basic idea is that a cyclone's area is defined by the
    isobar that surrounds only that cyclone and no SLP maxima.  Two minima can
    be part of the same cyclone if a) the ratio of their shared area size to the 
    size of the unshared area of the primary (lower pressure) minimum exceeds 
    the mcctol parameter and b) they are within mccdist from each other.
    
    field = the numpy array that you want to find the minima of.\n
    fieldCenters = field of cyclone center locations.\n
    centers = list of cyclone center objects.\n
    contint = contour interval used to define area.\n
    mcctol = the maximum ratio permitted between the unshared and total area in
        a multi-centered cyclone. "Unshared" area includes only the primary center. 
        "Shared" area is includes both the primary and secondary centers.\n
    mccdist = the maximum distance two minima can lie apart and still be 
        considered part of the same cyclone system.\n
    lats, longs = numpy arrays of latitude and longitude for the field.\n
    maxes = numpy array of field maxima; optional and set to 0 by default.
    '''
    if isinstance(maxes, int) == 1:
        field_max = np.zeros_like(field)
    else:
        field_max = maxes
    
    # Prepare preliminary outputs:
    cycField = np.zeros_like(fieldCenters) # set up empty system field
    fieldCenters2 = fieldCenters.copy() # set up a field to modify center identification
    # Make cyclone objects for all centers
    cols, rows, vals, las, los = [], [], [], [], []
    cyclones = copy.deepcopy(centers)
    types = np.zeros_like(np.array(centers))
    
    # And helper lists for area detection:
    for center in centers:
            vals.append(center.p_cent)
            rows.append(center.y)
            cols.append(center.x)
            las.append(center.lat)
            los.append(center.long)
    
    # Create list of ids and of ids that have not been assigned yet
    ids = np.arange(len(vals))
    candids = np.where(types == 0)[0]
    
    #####################
    # Start Area Loop
    #####################
    while len([t for t in types if t == 0]) > 0:
        # Identify the center ID as the index of the lowest possible value
        ## that hasn't already been assigned to a cyclone:
        cid = ids[np.where((types == 0) & (np.array(vals) == np.min(np.array(vals)[np.where(types == 0)[0]])))][0]
                        
        nMins = 0 # set flag to 0
        nMaxs = 0 # set flag to 0
        cCI = vals[cid] # set the initial contouring value
        
        # Identify the number of minima  within the mccdist
        distTest = [haversine(las[cid],las[i],los[cid],los[i]) > mccdist for i in candids]
        ncands = len(distTest) - np.sum(distTest) # Number of centers w/n the mccdist
        icands = candids[np.where(np.array(distTest) == 0)] # IDs for centers w/n mccdist
        
        #########################
        # If No MCC Is Possible #
        #########################
        # If there's no other minima w/n the mccdist, use the simple method
        if ncands == 1:
            while nMins == 0 and nMaxs == 0 and cCI < np.nanmax(field): # keep increasing interval as long as only one minimum is detected
                #Increase contour interval
                cCI = cCI + contint
                
                # Define contiguous areas
                fieldCI = np.where(field < cCI, 1, 0)
                areas, nA = ndimage.measurements.label(fieldCI)
                
                # Test how many minima are within the area associated with the minimum of interest -  Limit to minima within that area
                nMins = np.sum( np.where((areas == areas[rows[cid],cols[cid]]) & (fieldCenters > 0), 1, 0) ) - 1 # Count the number of minima identified (besides the minimum of focus)
                
                # Test how many maxima are within the area associated with the minimum of interest - Limit to maxima within that area           
                nMaxs = np.sum( np.where((areas == areas[rows[cid],cols[cid]]) & (field_max == 1), 1, 0) ) # Count the number of maxima identified
            
            # Re-adjust the highest contour interval
            cCI = cCI - contint
        
        ######################
        # If MCC Is Possible #
        ######################
        else:
            cCIs, aids = [], []
            while nMins == 0 and nMaxs == 0 and cCI < np.nanmax(field): # keep increasing interval as long as only one minimum is detected
                #Increase contour interval
                cCI = cCI + contint
                
                # Define contiguous areas
                fieldCI = np.where(field < cCI, 1, 0)
                areas, nA = ndimage.measurements.label(fieldCI)
                
                # Test how many minima are within the area associated with the minimum of interest
                areaTest = np.where((areas == areas[rows[cid],cols[cid]]) & (fieldCenters > 0)) # Limit to minima within that area
                
                # Test how many maxima are within the area associated with the minimum of interest - Limit to maxima within that area
                nMaxs = np.sum( np.where((areas == areas[rows[cid],cols[cid]]) & (field_max == 1), 1, 0) ) # Count the number of maxima identified
                
                # Record the area and the ids of the minima encircled for each contint
                locSub = [(areaTest[0][ls],areaTest[1][ls]) for ls in range(areaTest[0].shape[0])]
                idsSub = [i for i in ids if ((rows[i],cols[i]) in locSub) & (i != cid)]                
                cCIs.append(cCI)
                aids.append(idsSub)
                # Record the number of minima w/n the area that are outside the mccdist
                nMins = len(aids[-1]) - np.sum([i in icands for i in aids[-1]])
                
            # If there are possible secondaries within mccdist, evaluate MCC possibilities
            # Also, only check if its possible for the number of contour intervals to exceed mcctol
            if (len(aids) > 1) and ( len(aids[-1]) > 1 or (len(aids[-1]) == 1 and nMins == 0) ):
                # For each minimum in the last aids position before breaking, make a contour interval test,
                aids, cCIs = aids[:-1], cCIs[:-1]
                ## Starting with the last center to be added (fewest instances w/n the area)
                nCIs = [np.sum([i in ii for ii in aids]) for i in aids[-1]]
                breaker = 0
                
                while breaker == 0 and len(nCIs) > 0:
                    ai = aids[-1][np.where(np.array(nCIs) == np.min(nCIs))[0][0]] # Find id
                    ai0 = np.where(np.array([ai in i for i in aids]) == 1)[0][0] # First contour WITH secondary
                    
                    numC1 = (cCIs[ai0] - contint - vals[cid]) # The number of contour intervals WITHOUT secondaries
                    numC2 = (cCIs[-1] - vals[cid]) # The number of contour intervals WITH secondaries
                    
                    # If including secondaries substantially increases the number of contours involved...
                    if (numC1 / numC2) < mcctol:
                        for i in aids[-1]: # Add all of the other minima at this level as secondaries
                            cyclones[i].add_parent(rows[cid],cols[cid],cid)
                            fieldCenters2[rows[i],cols[i]] = 2
                            cyclones[cid].secondary.append(centers[i].id)
                            cyclones[i].type = 2
                            types[i] = 2 # And change the type to secondary so it will no longer be considered
                        
                        # Force the loop to end; all remaining minima must also be secondaries
                        breaker = 1
                        
                    else:
                        cCIs = cCIs[:ai0]
                        aids = aids[:ai0]
                        nCIs = [np.sum([i in ii for ii in aids]) for i in aids[-1]]
                
                # Once secondaries are accoutned for, re-establish the highest contour
                cCI = cCIs[-1]
            
            # Otherwise, ignore such possibilities
            else:
                cCI = cCI - contint
        
        #########################
        # Final Area Assignment #
        #########################
        # Assign final contiguous areas:
        fieldF = np.where(field < (cCI), 1, 0)
        areasF, nAF = ndimage.measurements.label(fieldF)
        
        # And identify the area associated with the minimum of interest:
        area = np.where((areasF == areasF[rows[cid],cols[cid]]) & (areasF != 0),1,0)
        if np.nansum(area) == 0: # If there's no area already,
            area[rows[cid],cols[cid]] = 1 # Give it an area of 1 to match the location of the center
        
        cycField = cycField + area # And update the system field
        
        # Then assign characteristics to the cyclone:
        cyclones[cid].type = 1
        cyclones[cid].p_edge = cCI
        cyclones[cid].area = np.nansum(area)
        
        # Also assign those characteristics to its secondaries:
        for cid_s in cyclones[cid].secondary:
            cyclones[cid_s].p_edge = cCI
            cyclones[cid_s].area = np.nansum(area)
        
        # When complete, change type of cyclone and re-set secondary candidates
        types[cid] = 1
        candids = np.where(types == 0)[0]
        #print("ID: " + str(cid) + ", Row: " + str(rows[cid]) + ", Col: " + str(cols[cid]) + \
        #        ", Area:" + str(np.nansum(area)))

    return cycField.astype(np.uint8), fieldCenters2, cyclones

'''###########################
Calculate Cyclone Associated Precipitation
###########################'''
def findCAP(cyclones,areas,plsc,ptot,yDist,xDist,lats,lons,pMin=0.375,r=250000):
    '''Calculates the cyclone-associated precipitation for each cyclone in
    a cyclone field object for a particular time. Input precipitation fields 
    must have the same projection and grid cell size as the cyclone field.\n
    
    Required inputs:\n
    cyclones = list of cyclone objects for the instant in time under consideration.
        Typically, this will be a subset of cyclones (e.g., only primary cyclone 
        centers) from a cyclonefield object
    areas = an array of 1s and 0s, where 1 indicates grid cells within the area of a cyclone\n
    (Note that both cyclones and areas can be found in a cyclonefield object)\n
    plsc = large-scale precipitation field
    ptot = total precipitation field
    yDist, xDist = numpy arrays of distances between rows and columns. Only 
        necessary if calculating gradients, so the default is 0.\n
    lats, longs = numpy arrays of latitude and longitude for the field.\n
    cellarea = area of a grid cell (km^2)
    pMin = the minimum amount of large-scale precipitation required for 
        defining contiguous precipitation areas
    r = radius defining  minimum area around which to search for precipitation
        (this is in addition to any area defined as part of the cyclone by the
        algorithm) -- must be same units as grid cell size
    
    Returns a field of CAP, but also updates the precipitation values for each
    cyclone center in the cyclone field object.
    '''
    #############
    # PREP WORK #
    #############
    # Eliminate Non-Finite values
    ptot, plsc = np.where(np.isfinite(ptot) == 1,ptot,0), np.where(np.isfinite(plsc) == 1,plsc,0)
    
    # Add edges to the precipitation rasters
    cR, cC = areas.shape[0], areas.shape[1]
    pR, pC = plsc.shape[0], plsc.shape[1]
    
    plsc = np.hstack(( np.zeros( (cR,int((cC-pC)/2)) ) , \
    np.vstack(( np.zeros( (int((cR-pR)/2),pC) ), plsc ,np.zeros( (int((cR-pR)/2),pC) ) )), \
    np.zeros( (cR,int((cC-pC)/2)) ) ))
    
    ptot = np.hstack(( np.zeros( (cR,int((cC-pC)/2)) ) , \
    np.vstack(( np.zeros( (int((cR-pR)/2),pC) ), ptot ,np.zeros( (int((cR-pR)/2),pC) ) )), \
    np.zeros( (cR,int((cC-pC)/2)) ) ))
    
    # Identify large-scale precipitation regions
    pMasked = np.where(plsc >= pMin, 1, 0)
    pAreas, nP = ndimage.measurements.label(pMasked)
    cAreas, nC = ndimage.measurements.label(areas)
    aIDs = [c.areaID for c in cyclones]
    
    # Identify cyclone ids
    ids = np.array(range(len(cyclones)))

    # Create empty lists/arrays
    cInt = [[] for p in range(nP+1)]# To store the cyc ID for each precip region
    cPrecip = np.zeros((len(ids))) # To store the total precip for each cyclone center
    cPrecipArea = np.zeros((len(ids))) # To store the precip area for each cyclone center
    
    ######################
    # FIND INTERSECTIONS #
    ######################
    for i in ids: # For each center,
        # Identify corresponding area
        c = cyclones[i]
        cArea = np.where(cAreas == aIDs[i],1,0) # Calc'd area
        
        # number of cells in either cardinal direction of the center of the kernel
        k = np.int( np.ceil( r /  min(yDist[c.y,c.x],xDist[c.y,c.x]) ) )
        # number of cells in each row and column of the kernel
        kdiameter = k*2+1
    
        # for each cell, calculate x distances and y distances from the center
        kxdists = np.tile(np.arange(-k, k + 1), (kdiameter, 1)) * xDist[c.y,c.x]
        kydists = np.rot90(np.tile(np.arange(-k, k + 1), (kdiameter, 1)) * yDist[c.y,c.x])
        
        # Assign True/False based on distance from center
        kernel = np.sqrt( np.square(kxdists) + np.square(kydists) ) <= r
        
        # Modify cyclone area
        cArea[c.y-k:c.y+k+1,c.x-k:c.x+k+1] = \
        np.where(cArea[c.y-k:c.y+k+1,c.x-k:c.x+k+1] + kernel != 0, 1, 0) # Add a radius-based area
        
        # Find the intersecting precip areas
        pInt = np.unique(cArea*pAreas)[np.where(np.unique(cArea*pAreas) != 0)]
        
        # Assign cyc id to intersecting precip areas
        for pid in pInt:
            cInt[pid].append(i)
    
    # Identify unique intersecting precip areas
    pList = [p for p in range(nP+1) if len(cInt[p]) != 0]
    
    ####################
    # PARTITION PRECIP #
    ####################
    # For each intersecting precip area,
    for p in pList:
        # If it only intersects 1 center...
        if len(cInt[p]) == 1:
            # Assign all TOTAL precip to the cyclone
            pArea = np.where(pAreas == p,1,0)
            cPrecip[cInt[p][0]] += np.sum(ptot*pArea)
            cPrecipArea[cInt[p][0]] += np.sum(pArea)
        
        # If more than one cyclone center intersects, 
        else:  # Assign each grid cell to the closest cyclone area
            # Identify coordinates for this precipitation area
            pcoords = np.array(np.where(pAreas == p)).T
            
            # Identify the area indices
            aInt = [aIDs[a] for a in cInt[p]] 
            
            # Find the minimum distance between each point in the precipitation area 
            # and any point in each intersecting cyclone area
            distances = []
            for ai in aInt: # Looping through each cyclone area
                acoords = np.array(np.where(cAreas == ai)).T # Identifying coordinates for the cyclone area
                distances.append( cdist(pcoords,acoords).min(axis=1) ) # Identifying the minimum distance from each precip area coordinate
            
            # Identify which cyclone area contained the shortest distance for each point within the precip area
            closest_index = np.array(distances).argmin(axis=0)
            
            # Assign TOTAL precip to the closest cyclone
            for i,ai in enumerate(aInt):
                ci = cInt[p][np.where(np.array(aInt) == ai)[0][0]] # Cyclone index associated with that area index
                
                pcoords2 = pcoords[closest_index == i] # Subset to just the precip area coordinate for which this cyclone's area is closest
                
                cPrecip[ci] += ptot[pcoords2[:,0],pcoords2[:,1]].sum() # Sum all precip
                cPrecipArea[ci] += pcoords2.shape[0] # Add all points
    
    ##################
    # RECORD PRECIP #
    #################
    # Final assignment of precip to primary cyclones
    for i in ids:
        cyclones[i].precip = cPrecip[i]
        cyclones[i].precipArea = cPrecipArea[i]

    # Return CAP field
    return ptot*np.in1d(pAreas,np.array(pList)).reshape(pAreas.shape)

def findCAP2(cycfield,plsc,ptot,yDist,xDist,lats,longs,pMin,r=250000):
    '''Calculates the cyclone-associated precipitation for each cyclone in
    a cyclone field object for a particular time. Input precipitation fields 
    must have the same projection and grid cell size as the cyclone field.\n
    
    Required inputs:\n
    cycfield = a cyclone field object with cyclone centers and areas already 
        calculated (using the findCenters and findAreas functions).
    plsc = large-scale precipitation field
    ptot = total precipitation field
    yDist, xDist = numpy arrays of distances between rows and columns. Only 
        necessary if calculating gradients, so the default is 0.\n
    lats, longs = numpy arrays of latitude and longitude for the field.\n
    cellarea = area of a grid cell (km^2)
    pMin = the minimum amount of large-scale precipitation required for 
        defining contiguous precipitation areas; array with same dimensions as
        plsc and ptot\n
    r = radius defining  minimum area around which to search for precipitation
        (this is in addition to any area defined as part of the cyclone by the
        algorithm) -- must be same units as grid cell size
    
    Returns a field of CAP, but also updates the precipitation values for each
    cyclone center in the cyclone field object.
    '''
    #############
    # PREP WORK #
    #############
    # Eliminate Non-Finite values
    ptot, plsc = np.where(np.isfinite(ptot) == 1,ptot,0), np.where(np.isfinite(plsc) == 1,plsc,0)
    
    # Add edges to the precipitation rasters
    cR, cC = cycfield.fieldAreas.shape[0], cycfield.fieldAreas.shape[1]
    pR, pC = plsc.shape[0], plsc.shape[1]
    
    plsc = np.hstack(( np.zeros( (cR,(cC-pC)/2) ) , \
    np.vstack(( np.zeros( ((cR-pR)/2,pC) ), plsc ,np.zeros( ((cR-pR)/2,pC) ) )), \
    np.zeros( (cR,(cC-pC)/2) ) ))
    
    ptot = np.hstack(( np.zeros( (cR,(cC-pC)/2) ) , \
    np.vstack(( np.zeros( ((cR-pR)/2,pC) ), ptot ,np.zeros( ((cR-pR)/2,pC) ) )), \
    np.zeros( (cR,(cC-pC)/2) ) ))
    
    pMin = np.hstack(( np.zeros( (cR,(cC-pC)/2) ) , \
    np.vstack(( np.zeros( ((cR-pR)/2,pC) ), pMin ,np.zeros( ((cR-pR)/2,pC) ) )), \
    np.zeros( (cR,(cC-pC)/2) ) )) 
    
    # Identify large-scale precipitation regions
    pMasked = np.where(plsc >= pMin, 1, 0)
    pAreas, nP = ndimage.measurements.label(pMasked)
    cAreas, nC = ndimage.measurements.label(cycfield.fieldAreas)
    aIDs = [c.areaID for c in cycfield.cyclones]
    
    # Identify cyclone ids
    ids = np.array(range(len(cycfield.centerCount())))
    ids1 = ids[np.where(np.array(cycfield.centerType()) == 1)]
    ids2 = ids[np.where(np.array(cycfield.centerType()) == 2)]
    
    # Create empty lists/arrays
    cInt = [[] for p in range(nP+1)]# To store the cyc ID for each precip region
    cPrecip = np.zeros((len(ids))) # To store the total precip for each cyclone center
    cPrecipArea = np.zeros((len(ids))) # To store the precip area for each cyclone center
    
    ######################
    # FIND INTERSECTIONS #
    ######################
    for i in ids1: # For each PRIMARY center,
        # Identify corresponding area
        c = cycfield.cyclones[i]
        cArea = np.where(cAreas == aIDs[i],1,0) # Calc'd area
        
        try: # Add a radius-based area if possible
            k = np.ceil( r / min(yDist[c.y,c.x],xDist[c.y,c.x]) )
            kernel = np.zeros((k*2+1,k*2+1))
            for row in range(int(k*2+1)):
                for col in range(int(k*2+1)):
                    if r >= (((row-k)*yDist[c.y,c.x])**2 + ((col-k)*xDist[c.y,c.x])**2)**0.5:
                        kernel[row,col]=1
            
            cArea[c.y-k:c.y+k+1,c.x-k:c.x+k+1] = \
            np.where(cArea[c.y-k:c.y+k+1,c.x-k:c.x+k+1] + kernel != 0, 1, 0) # Add a radius-based area
            for ii in c.secondary: # Add any secondary radius-based areas
                cc = cycfield.cyclones[ii]
                cArea[cc.y-k:cc.y+k+1,cc.x-k:cc.x+k+1] = \
                np.where(cArea[cc.y-k:cc.y+k+1,cc.x-k:cc.x+k+1] + kernel != 0, 1, 0)
        except:
            continue
        
        # Find the intersecting precip areas
        pInt = np.unique(cArea*pAreas)[np.where(np.unique(cArea*pAreas) != 0)]
        
        # Assign cyc id to intersecting precip areas
        for pid in pInt:
            cInt[pid].append(i)
    
    # Identify unique intersecting precip areas
    pList = [p for p in range(nP+1) if len(cInt[p]) != 0]
    
    ####################
    # PARTITION PRECIP #
    ####################
    # For each intersecting precip area,
    for p in pList:
        # If it only intersects 1 center...
        if len(cInt[p]) == 1:
            # Assign all TOTAL precip to the cyclone
            pArea = np.where(pAreas == p,1,0)
            cPrecip[cInt[p][0]] += np.sum(ptot*pArea)
            cPrecipArea[cInt[p][0]] += np.sum(pArea)
        
        # If more than one cyclone center intersects, 
        else:
            # Assign each grid cell individually based on closest center
            plocs = np.where(pAreas == p) # Find grid cells for area
            
            for i in range(len(plocs[0])):
                # Find the closest area
                aInt = [aIDs[a] for a in cInt[p]]
                ai = findNearestArea((lats[plocs[0][i],plocs[1][i]],longs[plocs[0][i],plocs[1][i]]),cAreas,aInt,latlon=(lats,longs))
                ci = cInt[p][np.where(np.array(aInt) == ai)[0][0]]
                
                # Assign TOTAL precip to the cyclone
                cPrecip[ci] += ptot[plocs[0][i],plocs[1][i]]
                cPrecipArea[ci] += 1
    
    ##################
    # RECORD PRECIP #
    #################
    # Final assignment of precip to primary cyclones
    for i in ids1:
        cycfield.cyclones[i].precip = cPrecip[i]
        cycfield.cyclones[i].precipArea = cPrecipArea[i]
    
    for i in ids2:
        par = cycfield.cyclones[i].parent['id']
        cycfield.cyclones[i].precip = cPrecip[par]
        cycfield.cyclones[i].precipArea = cPrecipArea[par]
    
    # Return CAP field
    return ptot*np.in1d(pAreas,np.array(pList)).reshape(pAreas.shape)

'''###########################
Nullify Cyclone-Related Data in a Cyclone Center Track Instance (not Track-Related Data)
###########################'''
def nullifyCycloneTrackInstance(ctrack,time,ptid):
    '''This function operates on a cyclonetrack object. Given a particular 
    time, it will turn any row with the time in the main data frame 
    (ctrack.data) into a partially nullified row. Some values are left 
    unchanged (id, mcc, time, x, y, u, and v, and the event flags). Some values
    (area, centers, radius, type) will be set to 0. The cyclone-specifc values
    are set to np.nan.
    
    The reason for having this function is that sometimes it's desirable to
    track a center's position during a merge or split, but it's not appropriate
    to assign other characteristics.
    
    ctrack = a cyclonetrack object
    time = a time step (float) corresponding to a row in ctrack (usu. in days)
    ptid = the track into which it's merging
    '''
    ctrack.data.loc[ctrack.data.time == time, "area"] = 0
    ctrack.data.loc[ctrack.data.time == time, "centers"] = 0
    ctrack.data.loc[ctrack.data.time == time, "radius"] = 0
    ctrack.data.loc[ctrack.data.time == time, "type"] = 0
    
    ctrack.data.loc[ctrack.data.time == time, "DpDt"] = np.nan
    ctrack.data.loc[ctrack.data.time == time, "DsqP"] = np.nan
    ctrack.data.loc[ctrack.data.time == time, "depth"] = np.nan
    ctrack.data.loc[ctrack.data.time == time, "p_cent"] = np.nan
    ctrack.data.loc[ctrack.data.time == time, "p_edge"] = np.nan
    
    ctrack.data.loc[ctrack.data.time == time,"ptid"] = ptid
    
    return ctrack

'''###########################
Initialize Cyclone Center Tracks
###########################'''
def startTracks(cyclones):
    '''This function initializes a set of cyclone tracks from a common start date
    when given a list of cyclone objects. It it a necessary first step to tracking
    cyclones. Returns a list of cyclonetrack objects and an update list of cyclone
    objects with track ids.
    
    cyclones = a list of objects of the class minimum.
    '''
    ct = []
    for c,cyc in enumerate(cyclones):
        # Define Etype
        if cyc.type == 2:
            Etype = 1
            ptid = cyc.parent["id"]
        else:
            Etype = 3
            ptid = c
        
        # Create Track
        ct.append(cyclonetrack(cyc,c,Etype,ptid))
        
        # Assign IDs to cyclone objects
        cyc.tid = c
    
    return ct, cyclones

'''###########################
Track Cyclone Centers Between Two Time Steps
###########################'''
def trackCyclones(cfa,cfb,ctr,maxdist,red,tmstep):
    '''This function tracks cyclone centers and areas between two times when
    given the cyclone fields for those two times, a cyclone track object that 
    is updated through time 1, maximum cyclone propagation distance (meters), 
    a speed reduction parameter, and the time interval (hr). The result is an 
    updated cyclone track object and updated cyclone field object for time 2.
    
    Steps in function:
    1. Main Center Tracking
        1.1. Center Lysis
        1.2. Center Tracking
    2. Center Merges
    3. Centers Splits and Genesis
        2.2. Center Splits
        2.3. Center Genesis
    4. Area Merges & Splits (MCCs)
        4.1. Area Merges
        4.2. Special Area Lysis (Re-Genesis)
        4.3. Area Splits
    5. Clean-Up for MCCs
    '''
    # First make copies of the cyclonetrack and cyclonefield inputs to ensure that nothing is overwritten
    ct = copy.deepcopy(ctr)
    cf1 = copy.deepcopy(cfa)
    cf2 = copy.deepcopy(cfb)
    
    # Calculate the maximum number of cells a cyclone can move
    time1 = cf1.time
    time2 = cf2.time

    # Create helper lists:
    y2s = cf2.lats()
    x2s = cf2.longs()
    mc2 = [[] for i in y2s] # stores ids of cf1 centers that map to each cf2 center
    sc2 = [[] for i in y2s] # stores ids of cf1 centers that were within distance to map but chose a closer cf2 center
    sc2dist = [[] for i in y2s] # stores the distance between the cf1 centers and rejected cf2 centers
    
    y1s, x1s = cf1.lats(), cf1.longs() # store the locations from time 1
    p1s = list(np.array(cf1.p_edge()) - np.array(cf1.p_cent())) # store the depth from time 1
    
    ################################
    # PART 1. MAIN CENTER TRACKING #
    ################################
    # Loop through each center in cf1 to find matches in cf2
    for c,cyc in enumerate(cf1.cyclones):
        cyct = ct[cyc.tid] # Link the cyclone instance to its track
        
        # Create a first guess for the next location of the cyclone:
        if len(cyct.data) == 1: # If cf1 represents the genesis event, first guess is no movement
            latq = cyc.lat
            longq = cyc.long
        elif len(cyct.data) > 1: # If the cyclone has moved in the past, a linear projection of its movement is the best guess
            latq = addLat(cyc.lat,dist2lat(float(red*cyct.data.loc[cyct.data.time == cyc.time,"v"])*tmstep))
            longq = addLong(cyc.long,dist2long(float(red*cyct.data.loc[cyct.data.time == cyc.time,"u"])*tmstep,cyc.lat,cyc.lat))
        
        # Test every point in cf2 to see if it's within distance d of both (yq,xq) AND (y,x)
        pdqs = [haversine(y2s[p],latq,x2s[p],longq) for p in range(len(y2s))]
        pds = [haversine(y2s[p],cyc.lat,x2s[p],cyc.long) for p in range(len(y2s))]
        
        pds_n = sum([((pds[p] <= maxdist) and (pdqs[p] <= maxdist)) for p in range(len(pdqs))])
        
        ##########################
        # PART 1.1. CENTER LYSIS #
        # If no point fits the criterion, then the cyclone experienced lysis
        if pds_n == 0:
            # Add a nullified instance for time2 at the same location as time 1
            cyc_copy = copy.deepcopy(cyc)
            cyc_copy.time = time2
            cyct.addInstance(cyc_copy)
            cyct = nullifyCycloneTrackInstance(cyct,time2,cyc.tid)
            # Add a lysis event
            cyct.addEvent(cyc,time2,"ly",3)
        
        ##############################
        # PART 1.2. CENTER TRACKING #
        # If one or more points fit the criterion, select the nearest neighbor to the projection
        else:
            # Take the one closest to the projected point (yq xq)
            c2 = np.where(np.array(pdqs) == min(pdqs))[0][0]
            cyct.addInstance(cf2.cyclones[c2])
            cf2.cyclones[c2].tid = cyc.tid # link the two cyclone centers with a track id
            mc2[c2].append(c) # append cf1 id to the merge list
            
            # Remove that cf2 cyclone from consideration
            pds[c2] = maxdist+1
            pds_n = pds_n - 1
            
            # Add the cf1 center to the splits list for the remaining cf2 centers that fit the dist criteria
            while pds_n > 0:
                s2 = np.where(np.array(pds) == min(pds))[0][0]
                sc2[s2].append(c)
                sc2dist[s2].append(min(pds))
                pds[s2] = maxdist+1
                pds_n = pds_n -1
    
    ##########################
    # PART 2. CENTER MERGES #
    #########################
    # First, remove the split possibility for any time 2 center that is continued
    for id2 in range(len(sc2)):
        if (len(sc2[id2]) > 0) and (len(mc2[id2]) > 0):
            sc2[id2], sc2dist[id2] = [], []
    
    # Check for center merges (mc) and center splits (sc)
    for id2 in range(len(mc2)):
        # If there is only one entry in the merge list, then it's a simple tracking; nothing else required
        # But if multiple cyclones from cf1 match the same cyclone from cf2 -> CENTER MERGE
        if len(mc2[id2]) > 1:
            ### DETERMINE PRIMARY TRACK ###
            dist1_mc2 = [haversine(y1s[i],y2s[id2],x1s[i],x2s[id2]) for i in mc2[id2]]
            p1s_mc2 = [p1s[i] for i in mc2[id2]]
            tid_mc2 = [cf1.cyclones[i].tid for i in mc2[id2]]
            
            # Select the closer, then deeper storm for continuation
            
            # Which is closest?
            dist1min = np.where(np.array(dist1_mc2) == min(dist1_mc2))[0]
            if len(dist1min) == 1: # if one center is closest
                id1 = mc2[id2][dist1min[0]] # find the id of the closest
            # Which center is deepest?
            else: # if multiple centers are same distance, choose the greater depth
                id1 = mc2[id2][np.where(np.array(p1s_mc2) == max(p1s_mc2))[0][0]] # find id of the max cf1 depth
                # Note that if two cyclones have the same distance & depth, the first by id is automatically taken.
            
            # Check if ptid of id1 center is the tid of another merge candidate
            ptid1 = int(ct[cf1.cyclones[id1].tid].data.loc[ct[cf1.cyclones[id1].tid].data.time == cf1.time,"ptid"])
            ptid_test = [ptid1 == i and i != cf1.cyclones[id1].tid for i in tid_mc2]
            if sum(ptid_test) > 0: # if the ptid is the tid of one of the other candidates, merge into the parent instead
                id1 = int(ct[ptid1].data.loc[ct[ptid1].data.time == time1,"id"])
            
            # Assign the primary track tid
            cf2.cyclones[id2].tid = cf1.cyclones[id1].tid
            
            ### DEAL WITH NON-CONTINUED CENTERS ###
            c1extra = copy.deepcopy(mc2[id2])
            c1extra.remove(id1)
            
            for c1id in c1extra:
                # First check if this center could have continued to another center in time 2
                sptest = [c1id in sc for sc in sc2]
                # If yes, then change the track assignment
                if np.sum(sptest) > 0:
                    # Find the new closest center
                    sc2s = np.where(np.array(sptest) == 1)[0] # The ids for the candidate time 2 centers
                    sc2dists = []
                    for sc2id in sc2s:
                        sc2dists.append( sc2dist[sc2id][np.where(np.array(sc2[sc2id]) == c1id)[0][0]] )
                    
                    c2id = sc2s[np.where(np.array(sc2dists) == np.min(sc2dists))][0] # New time 2 id for continuation of the time 1 track
                    
                    # Remove the original instance established in Part 1.2.
                    ct[cf1.cyclones[c1id].tid].removeInstance(time2)
                    
                    # And add a new instance
                    ct[cf1.cyclones[c1id].tid].addInstance(cf2.cyclones[c2id])
                    cf2.cyclones[c2id].tid = cf1.cyclones[c1id].tid # link the two cyclone centers with a track id
                    mc2[c2id].append(c1id) # append cf1 id to the merge list
                    mc2[id2].remove(c1id) # remove from its former location
                    sc2[c2id] = [] # clear the corresponding element of the split list
                
                # Otherwise, assign a merge event with the primary track
                else:
                    # If the two centers shared the same parent in time 1 (they were an mcc), it's just a center merge
                    if cf1.cyclones[c1id].parent["id"] == cf1.cyclones[id1].parent["id"]:
                        # Add merge event to primary track
                        ct[cf1.cyclones[id1].tid].addEvent(cf2.cyclones[id2],time2,"mg",1,otid=cf1.cyclones[c1id].tid)
                        
                        # Add merge event to non-continued (secondary) track
                        ct[cf1.cyclones[c1id].tid].addEvent(cf2.cyclones[id2],time2,"mg",1,otid=cf1.cyclones[id1].tid)
                        ct[cf1.cyclones[c1id].tid].addEvent(cf2.cyclones[id2],time2,"ly",1) # Add a lysis event
                        # Set stats to zero for current time since I really just want the (y,x) location
                        ct[cf1.cyclones[c1id].tid] = nullifyCycloneTrackInstance(ct[cf1.cyclones[c1id].tid],time2,cf1.cyclones[id1].tid)
                    
                    # If they had different parents, it's both a center merge and an area merge
                    else:
                        # Add merge event o primary track
                        ct[cf1.cyclones[id1].tid].addEvent(cf2.cyclones[id2],time2,"mg",3,otid=cf1.cyclones[c1id].tid)
                        
                        # Add merge event to non-continued (secondary) track
                        ct[cf1.cyclones[c1id].tid].addEvent(cf2.cyclones[id2],time2,"mg",3,otid=cf1.cyclones[id1].tid)
                        ct[cf1.cyclones[c1id].tid].addEvent(cf2.cyclones[id2],time2,"ly",3) # Add a lysis event
                        # Set stats to zero for current time since I really just want the (y,x) location
                        ct[cf1.cyclones[c1id].tid] = nullifyCycloneTrackInstance(ct[cf1.cyclones[c1id].tid],time2,cf1.cyclones[id1].tid)
    
    #####################################
    # PART 3. CENTER SPLITS AND GENESIS #
    #####################################
    for id2 in range(len(sc2)):
        ###########################
        # PART 3.1. CENTER SPLITS #
        # If no cyclones from cf1 match a particular cf2 cyclone, it's either a center split or a pure genesis event
        if len(mc2[id2]) == 0 and len(sc2[id2]) > 0: # if there's one or more centers that could have tracked there -> SPLIT
            ### DETERMINE SOURCE CENTER ###
            # Make the split point the closest center of the candidate(s)
            dist_sc2 = [haversine(cf1.cyclones[i].lat,cf2.cyclones[id2].lat,cf1.cyclones[i].long,cf2.cyclones[id2].long) for i in sc2[id2]]
            id1= sc2[id2][np.where(np.array(dist_sc2) == min(dist_sc2))[0][0]]
            
            # Start the new track a time step earlier
            cf2.cyclones[id2].tid = len(ct) # Assign a new track id to the cf2 center
            ct.append(cyclonetrack(cf1.cyclones[id1],tid=cf2.cyclones[id2].tid)) # Make a new track with that id
            # Set stats to zero since I really just want the (y,x) location
            ct[cf2.cyclones[id2].tid] = nullifyCycloneTrackInstance(ct[cf2.cyclones[id2].tid],time1,cf1.cyclones[id1].tid)
            
            # Add an instance for the current time step (cf2)
            ct[cf2.cyclones[id2].tid].addInstance(cf2.cyclones[id2])
            # Adjust when the genesis is recorded
            ct[cf2.cyclones[id2].tid].events.time = time2
            ct[cf2.cyclones[id2].tid].data.loc[ct[cf2.cyclones[id2].tid].data.time == time1,"Ege"] = 0
            ct[cf2.cyclones[id2].tid].data.loc[ct[cf2.cyclones[id2].tid].data.time == time2,"Ege"] = 3
            
            ### ASSIGN SPLIT EVENTS ###
            # Find the id of the other branch of the split in the time of cf2
            id2_1 = int(ct[cf1.cyclones[id1].tid].data.loc[ct[cf1.cyclones[id1].tid].data.time == time2,"id"])
            
            # Add a split event to the primary track and the new track
            if cf2.cyclones[id2_1].parent["id"] == cf2.cyclones[id2].parent["id"]:
                # If the two centers have the same parent in time 2, it was just a center split
                ct[cf1.cyclones[id1].tid].addEvent(cf2.cyclones[id2_1],time2,"sp",1,otid=cf2.cyclones[id2].tid)
                ct[cf2.cyclones[id2].tid].addEvent(cf2.cyclones[id2],time2,"sp",1,otid=cf1.cyclones[id1].tid)
                ct[cf2.cyclones[id2].tid].data.loc[ct[cf2.cyclones[id2].tid].data.time == time2,"Ege"] = 1
                ct[cf2.cyclones[id2].tid].events.loc[(ct[cf2.cyclones[id2].tid].events.time == time2) & \
                    (ct[cf2.cyclones[id2].tid].events.event == "ge"),"Etype"] = 1
            else:
                # If they don't, then it was also an area split
                ct[cf1.cyclones[id1].tid].addEvent(cf2.cyclones[id2_1],time2,"sp",3,otid=cf2.cyclones[id2].tid)
                ct[cf2.cyclones[id2].tid].addEvent(cf2.cyclones[id2],time2,"sp",3,otid=cf1.cyclones[id1].tid)
                # Amend the parent track id to be it's own track now that it's split
                ct[cf2.cyclones[id2].tid].data.loc[ct[cf2.cyclones[id2].tid].data.time == time2,"ptid"] = cf2.cyclones[id2].tid
                ct[cf2.cyclones[id2].tid].ptid = cf2.cyclones[id2].tid
                # Note that this might be overwritten yet again if this center has an area merge with another center
                
        ############################
        # PART 3.2. CENTER GENESIS #
        elif len(mc2[id2]) == 0 and len(sc2[id2]) == 0: # if there's no center that could have tracked here -> GENESIS
            cf2.cyclones[id2].tid = len(ct) # Assign the track id to the cf2 center
            
            if cf2.cyclones[id2].centerCount() == 1: # If it's a scc, it's both an area and center genesis
                ct.append(cyclonetrack(cf2.cyclones[id2],cf2.cyclones[id2].tid,3,cf2.cyclones[id2].tid)) # Make a new track
            
            else: # If it's a mcc, things are more complicated
                # Find center ids of mcc centers
                mcc_ids = cf2.cyclones[cf2.cyclones[id2].parent["id"]].secondary
                mcc_ids.append(cf2.cyclones[id2].parent["id"])
                # Find which have prior tracks
                prior = [( (len(mc2[mccid]) > 0) or (len(sc2[mccid]) > 0) ) for mccid in mcc_ids]
                
                if (sum(prior) > 0) or (cf2.cyclones[id2].type == 2): 
                    # If it's a secondary center or if one of the centers in the mcc already has a track, 
                    ### then it's only a center genesis
                    
                    ct.append(cyclonetrack(cf2.cyclones[id2],cf2.cyclones[id2].tid,1)) # Make a new track
                else:
                    # If all centers in the mcc are new and this is the primary center, 
                    ### then it's both center and area genesis
                    ct.append(cyclonetrack(cf2.cyclones[id2],cf2.cyclones[id2].tid,3,cf2.cyclones[id2].tid)) # Make a new track
    
    #######################################################
    # PART 4. AREA-ONLY SPLITS & MERGES and SPECIAL LYSIS #
    #######################################################
    ########################
    # PART 4.1. AREA MERGE #
    for cy2 in cf2.cyclones:
        # If cy2 is part of an mcc
        if int(ct[cy2.tid].data.loc[ct[cy2.tid].data.time == time2,"centers"]) != 1:
            # Find the id of cy2 in time of cf1:
            if len(ct[cy2.tid].data.loc[(ct[cy2.tid].data.time == time1) & (ct[cy2.tid].data.type != 0)]) == 0:
                continue # gensis event, no center to compare
            else:
                cy2_id1 = int(ct[cy2.tid].data.loc[ct[cy2.tid].data.time == time1,"id"])
                
                # Find track ids for each center in the mcc that isn't cy2
                ma_tids = [cy.tid for cy in cf2.cyclones if ((cy.parent["id"] == cy2.parent["id"]) and (cy.id != cy2.id))]
                
                for ti in  ma_tids: # for each track id
                    if len(ct[ti].data.loc[ct[ti].data.time == time1,"id"]) == 0:
                        continue # genesis event, no center to compare
                    else: 
                        ma_id1 = int(ct[ti].data.loc[ct[ti].data.time == time1,"id"] )# Find the center id for cf1
                        ma_parent_id = cf1.cyclones[ma_id1].parent["id"] # Find parent id at time of cf1
                        if ma_parent_id != cf1.cyclones[cy2_id1].parent["id"]: # If the two are not the same
                            # Assign an area merge to cy2's track
                            ct[cy2.tid].addEvent(cy2,time2,"mg",2,otid=ti)
    
    for cy1 in cf1.cyclones: 
        # If cy1 was part of an mcc
        if int(ct[cy1.tid].data.loc[ct[cy1.tid].data.time == time1,"centers"]) != 1:
            # What was the primary track in time 1?
            ptid1 = int(ct[cy1.tid].data.loc[ct[cy1.tid].data.time == time1,"ptid"])
            
            ###########################
            # PART 4.2. SPECIAL LYSIS #
            if len(ct[cy1.tid].data.loc[(ct[cy1.tid].data.time == time2) & (ct[cy1.tid].data.type != 0)]) == 0:
                # If cy1 was the system center...
                if cy1.tid == ptid1:
                    # Did any other centers survive?
                    mcc_ids = copy.deepcopy(cf1.cyclones[cy1.parent["id"]].secondary)
                    mcc_ids.append(cy1.parent["id"])
                    survtest = [len(ct[cf1.cyclones[i].tid].data.loc[(ct[cf1.cyclones[i].tid].data.time == time2) \
                        & (ct[cf1.cyclones[i].tid].data.type != 0)]) > 0 for i in mcc_ids]
                    
                    # If another center did survive, then there is cyclone re-genesis
                    if sum(survtest) > 0:
                        # Find the deepest surviving center
                        mcc_ids_surv = [mcc_ids[i] for i in range(len(mcc_ids)) if survtest[i] == 1]
                        mcc_ids_surv_p = [float(ct[cf1.cyclones[i].tid].data.loc[ct[cf1.cyclones[i].tid].data.time == time2,"p_cent"]) \
                                            for i in mcc_ids_surv]
                        reid1 = mcc_ids_surv[np.where(np.array(mcc_ids_surv_p) == min(mcc_ids_surv_p))[0][0]]
                        # Find the cf2 version of that center
                        reid2 = int(ct[cf1.cyclones[reid1].tid].data.loc[ct[cf1.cyclones[reid1].tid].data.time == time2,"id"])
                        # Assign it an area regenesis event & make it the primary track
                        ct[cf2.cyclones[reid2].tid].addEvent(cf2.cyclones[reid2],time2,"ge",2)
                        ct[cf2.cyclones[reid2].tid].addEvent(cf2.cyclones[reid2],time2,"rg",2,otid=cy1.tid)
                        ct[cf2.cyclones[reid2].tid].data.loc[ct[cf2.cyclones[reid2].tid].data.time == time2,"ptid"] = cf2.cyclones[reid2].tid
                        ct[cf2.cyclones[reid2].tid].ptid = cf2.cyclones[reid2].tid
                    # If no other centers survived, then it's just a normal type 3 lysis --> no change
                    else:
                        continue
                
                else: # cy1 was NOT the system center...
                    # Then it's just a secondary lysis event --> change event type to 1
                    ct[cy1.tid].data.loc[ct[cy1.tid].data.time == time2,"Ely"] = 1
                    ct[cy1.tid].events.loc[(ct[cy1.tid].events.time == time2) & (ct[cy1.tid].events.event == "ly"),"Etype"] = 1
            
            ########################
            # PART 4.3. AREA SPLIT #
            else: 
                # Otherwise, find the id of cy1 in time of cf2:
                c1_id2 = int(ct[cy1.tid].data.loc[ct[cy1.tid].data.time == time2,"id"])
                
                # Find track ids for each center in the mcc that isn't cy1
                sa_tids = [cy.tid for cy in cf1.cyclones if ((cy.parent["id"] == cy1.parent["id"]) and (cy.id != cy1.id))]
                
                for ti in  sa_tids: # for each track id
                    if len(ct[ti].data.loc[(ct[ti].data.time == time2) & (ct[ti].data.type != 0),"id"]) == 0:
                        continue # This was a lysis event, no center to compare
                    
                    else: 
                        sa_id2 = int(ct[ti].data.loc[ct[ti].data.time == time2,"id"] )# Find the center id for cf2
                        sa_parent_id = cf2.cyclones[sa_id2].parent["id"] # Find parent id at time of cf2
                        if sa_parent_id != cf2.cyclones[c1_id2].parent["id"]: # If the two are not the same
                            # Assign an area split to cy1's track
                            ct[cy1.tid].addEvent(cf2.cyclones[c1_id2],time2,"sp",2,otid=ti)
                            if ptid1 != cy1.tid: # If cy1 was NOT the system center
                                # Then change the track id to its own now that it has split
                                ct[cy1.tid].data.loc[ct[cy1.tid].data.time == time2,"ptid"] = cy1.tid
                                ct[cy1.tid].ptid = cy1.tid
                                # And add an area genesis event
                                ct[cy1.tid].addEvent(cf2.cyclones[c1_id2],time2,"ge",2)
    
    ##########################################################
    # PART 5. UPDATE ptid OF MULTI-CENTER CYCLONES IN TIME 2 #  
    ##########################################################    
    for cy2 in cf2.cyclones:
        # Identify MCCs by the primary center
        if int(ct[cy2.tid].data.loc[ct[cy2.tid].data.time == time2,"centers"]) > 1:
            # For each mcc, identify the cycs and their tids for each center
            mcy2s = [cy for cy in cf2.cyclones if cy.parent["id"] == cy2.parent["id"]]
            mtids = [cy.tid for cy in cf2.cyclones if cy.parent["id"] == cy2.parent["id"]]
            
            # Grab the depth and lifespan at time 2:
            mp2s = [cy.p_edge - cy.p_cent for cy in mcy2s]
            
            # Which tracks also existed in cf1? (excludes split genesis markers)
            pr_mtids = [ti for ti in mtids if len(ct[ti].data.loc[ct[ti].data.time == time1,"type"]) > 0 \
                    and int(ct[ti].data.loc[ct[ti].data.time == time1,"type"]) > 0]
            
            # If none of the tracks existed in cf1 time, 
            if len(pr_mtids) == 0:
                # then choose the deepest in cf2 time as ptid
                ptid2 = mtids[np.where(np.array(mp2s) == max(mp2s))[0][0]]
            
            # If only one track existed in cf1 time,
            elif len(pr_mtids) == 1:
                # Use it's tid as the ptid for everything
                ptid2 = int(ct[pr_mtids[0]].tid)
            
            # If more than one track existed in cf1 time,
            else:
                # Assign the center with the greatest depth
                mp2s_pr = [float(ct[ti].data.loc[ct[ti].data.time == time2,"depth"]) for ti in pr_mtids]
                tid2 = mtids[np.where(np.array(mp2s) == max(mp2s_pr))[0][0]] # find id of the max cf2 depth
                # Note that if two cyclones have the same depth, the first by id is automatically taken.
                try:
                    ptid2 = int(ct[tid2].data.loc[ct[tid2].data.time == time2,"ptid"]) # identify its ptid as ptid for system
                except: 
                    ptid2 = tid2 # if it has no ptid, then assign its tid as ptid
            
            # Loop through all centers in the mcc
            for mtid in mtids:
                # Assign the ptid to all centers in the mcc
                ct[mtid].data.loc[ct[mtid].data.time == time2,"ptid"] = ptid2
                ct[mtid].ptid = ptid2
                
                # Add area lysis events to any non-ptid tracks that experienced an area merge
                if ct[mtid].tid != ptid2 and int(ct[mtid].data.loc[ct[mtid].data.time == time2,"Emg"]) == 2:
                    ct[mtid].addEvent(cf2.cyclones[int(ct[mtid].data.loc[ct[mtid].data.time == time2,"id"])],time2,"ly",2,otid=ptid2)
        
        # For center-only lysis events, set the final ptid to match the last observed ptid
        if int(ct[cy2.tid].data.loc[ct[cy2.tid].data.time == time2,"Ely"]) == 1:
            ct[cy2.tid].data.loc[ct[cy2.tid].data.time == time2,"ptid"] = int(ct[cy2.tid].data.loc[ct[cy2.tid].data.time == time1,"ptid"])
            ct[cy2.tid].ptid = int(ct[cy2.tid].data.loc[ct[cy2.tid].data.time == time1,"ptid"])
    
    if len(cf2.tid()) != len(list(set(cf2.tid()))):
        raise Exception("Number of centers in cf2 does not match the number of \
        tracks assigned. Multiple centers may have been assigned to the same track.")
    
    return ct, cf2 # Return an updated cyclonetrack object (corresponds to ctr)
    ### and cyclonefield object for time 2 (corresponds to cfb)

'''###########################
Split Tracks into Active and Inactive
###########################'''
def splitActiveTracks(ct,cf):
    '''Given a list of cyclone tracks, this function creates two new lists: one
    with all inactive tracks (tracks that have already experienced lysis) and
    one with all active tracks (tracks that have not experienced lysis). Next,
    the track ids (and parent track ids) are reset for all active tracks and 
    the related cyclone field.
    
    ct = list of cyclone tracks
    cf = cyclone field object
    
    Returns: ([active tracks], [inactive tracks])
    (The cyclone field object is mutable and automatically edited.)
    '''
    ct_inactive, ct_active, tid_active = [], [], [] # Create empty lists
    # Sort tracks into active and inactive
    for track in ct:
        if track != 0 and ( (1 in list(track.data.Ely)) or (3 in list(track.data.Ely)) ):
            ct_inactive.append(track)
        else:
            ct_active.append(track)
            tid_active.append(track.tid)
    
    # Reformat tids for active tracks
    tid_activeA = np.array(tid_active)
    for tr in range(len(ct_active)):
        ct_active[tr].tid = tr
        ct_active[tr].ftid = tid_active[tr]
        if ct_active[tr].ptid in tid_active:
            ct_active[tr].ptid = int(np.where(tid_activeA == ct_active[tr].ptid)[0][0])
    
    for cyctr in cf.cyclones:
        if cyctr.tid in tid_active:
            cyctr.tid = int(np.where(tid_activeA == cyctr.tid)[0][0])
        else:
            cyctr.tid = np.nan # These are inactive tracks, so they don't matter anymore.
    
    return ct_active, ct_inactive

'''###########################
Realign Track IDs for Cyclone Tracks and Cyclone Field
###########################'''
def realignPriorTID(ct,cf1):
    '''This is a very specific function used to realign the track ids for a 
    list of active cyclone tracks and a cyclone field object. The cyclone field 
    object must correspond to the final time step recorded in the track 
    objects. Additionally, the cyclone field object must have the same number 
    of centers as the track list has tracks.
    
    Inputs:
    ct = a list of cyclone track objects
    cf1 = a cyclone field object corresponding to the final time recorded in ct
    
    Output:
    no return, but the tids in cf1 are modified.
    '''
    
    if len(ct) != len(cf1.cyclones):
        raise Exception("Number of tracks in ct doesn't equal the number of centers in cf1.")
    if (cf1.time != ct[0].data.time.iloc[-1]) or (cf1.time != ct[-1].data.time.iloc[-1]):
        raise Exception("The time for cf1 is not the final time recorded for ct.")
    else:
        for tid in range(len(ct)): # For each track
            cid = int(ct[tid].data.loc[ct[tid].data.time == cf1.time,"id"]) # Find the center id for the final time step
            cf1.cyclones[cid].tid = tid # Reset the tid for the corresponding center in cf1

'''###########################
Convert Cyclone Center Tracks to Cyclone System Tracks
###########################'''
def cTrack2sTrack(ct,cs0=[],dateref=[1900,1,1,0,0,0],rg=0,lyb=1,dpy=365):
    '''Cyclone tracking using the trackCyclones function is performed on 
    cyclone centers, including secondary centers.  But since the primary
    center of a cyclone at timestep 1 might not share the same track as the 
    primary center of the same cyclone at timestep 2, it may sometimes be 
    desirable to follow only that primary center and so receive a system-based
    view of tracks. This function performs such a conversion post-tracking,
    ensuring that the track of a system always follows the primary center of
    that system. All other tracks are then built around that idea.
    
    ct = a list of cyclonetrack objects
    cs0 = a list of systemtrack objects from the prior month (defaults to an
    empty list, meaning there is no prior month)
    dateref = the reference date to use for determining the month
    rg = boolean; if set to 1, then a system track will be extended if one of 
    the secondary centers continues (a regenesis event); if set to 0, the re-
    genesis will be ignored and the surviving secondary will be treated as a 
    new system. Defaults to 0.
    
    Two lists of cyclone system objects are returned: one for the current month 
    (ct -> cs) and an updated one for the prior month (cs0 -> cs0)
    '''
    # Define month
    mt = timeAdd(dateref,[0,0,list(ct[0].data.time)[-1],0,0,0])
    mt[2], mt[3], mt[4], mt[5] = 1, 0, 0, 0
    days = daysBetweenDates(dateref,mt,lyb,dpy)
    
    cs = []
    
    # STEP 1: LIMIT TO PTID (primary tracks)
    for t in ct: # Loop through each original tracks
        ptidtest = [t.tid != p for p in t.data.loc[(t.data.type != 0) & (t.data.time >= days),"ptid"]] + \
            [t.ftid != p for p in t.data.loc[(t.data.type != 0) & (t.data.time < days),"ptid"]] # Current Month \ Prior Month
        if sum(ptidtest) == 0: # If ptid always equals tid
            cs.append(systemtrack(t.data,t.events,t.tid,len(cs),t.ftid)) # Append to system track list
        
        # If ptid is never equal to tid, the track is always secondary, so ignore it
        # But if the ptid is sometimes equal to the tid, the track needs to be split up
        elif sum(ptidtest) < len(t.data.loc[t.data.type != 0]): 
            # Start empty data frames for data and events:
            data = pd.DataFrame()
            events = pd.DataFrame()
            # Observe each time...
            
            for r in t.data.time:
                rdata = t.data.loc[t.data.time == r] # Pull out the row for this time
                
                # If the track is indepedent at this time step:
                if ( (t.tid == int(rdata["ptid"])) or (t.ftid == int(rdata["ptid"])) ) and ( int(rdata["type"]) != 0 ):
                    # Append the row to the open system
                    data = data.append(rdata, ignore_index=1, sort=1)
                    events = events.append(t.events.loc[t.events.time == r], ignore_index=1, sort=1)
                
                elif len(data) > 0:
                    # Append the row to the open system
                    data = data.append(rdata, ignore_index=1, sort=1)
                    events = events.append(t.events.loc[t.events.time == r], ignore_index=1, sort=1)
                    # Close the system by adding it to the cs list
                    cs.append(systemtrack(data,events,t.tid,len(cs),t.ftid))
                    # Nullify the final step 
                    nullifyCycloneTrackInstance(cs[-1],r,data.loc[data.time == r,"ptid"])
                    # Create a new open system:
                    data = pd.DataFrame()
                    events = pd.DataFrame()
            
            # After last is reached, end the open system if it has any rows
            if len(data) > 0:
                # Add any lysis events if they exist
                events = events.append(t.events.loc[t.events.time == r+(t.data.time.iloc[1] - t.data.time.iloc[0])], ignore_index=1, sort=1)
                # Append to cs list
                cs.append(systemtrack(data,events,t.tid,len(cs),t.ftid))
    
    # STEP 2: COMBINE REGENESIS CASES  
    cs = np.array(cs)
    if rg == 1:
        sys_tids = np.array([ccc.tid for ccc in cs])
        
        # CASE 1: Dead Track in Prior Month, Regenerated Track in This Month       
        # Identify the track id of regenerated tracks that died in prior month
        rg_otids, rg_tids, dels = [], [], []
        for t in cs:
            rgs = np.sum(t.events.loc[t.events.time < days,"event"] == "rg")
            if rgs > 0:
                rg_tids.append(t.tid)
                rg_otids.append(int(t.events.loc[t.events.event == "rg","otid"]))
        
        otids = np.array([aa.tid for aa in cs0])
        for o in range(len(rg_otids)): # For each dead track
            # Note the position of the dead track object
            delids = np.where(otids == rg_otids[o])[0]
            try: # Try linking to the prior month's track if possible
                dels.append(delids[-1])
                
                # Extract the dead track objects
                tDead = cs0[dels[o]] # Def of regenesis requires that primary track has experience type 3 lysis
                
                # Extract the regenerated track object
                sid_cands = np.where(sys_tids == rg_tids[o])[0] # Candidate track objects
                sid_rgcode = np.array([ np.sum(cs[sidc].data.Erg == 2) > 0 and \
                    (cs[sidc].data.loc[cs[sidc].data.Erg == 2,"time"].iloc[0] == tDead.data.time.iloc[-1]) \
                     for sidc in sid_cands ]) # Does this track have a regeneration?
                
                sid = sid_cands[np.where(sid_rgcode == 1)[0][0]] # sid of the regenerated track
                tRegen = cs[sid]
                
                # Splice together with the regenerated track
                cs[sid].data = tDead.data[:-1].append(tRegen.data.loc[tRegen.data.time >= tDead.data.time.iloc[-1]], ignore_index=1, sort=1)
                cs[sid].events = tDead.events[:-1].append(tRegen.events.loc[(tRegen.events.time >= tDead.data.time.iloc[-1])], ignore_index=1, sort=1)
                cs[sid].data.loc[cs[sid].data.Erg > 0,"Ely"] = 0
                cs[sid].data.loc[cs[sid].data.Erg > 0,"Ege"] = 0
            except:
                continue
        
        # CLEAN UP
        # Remove the dead tracks from the current month
        cs0 = np.delete(cs0,dels)
        
        # CASE 2: Dead Track and Regenerated Track in Same Month
        # Identify the track id of regenerated tracks that died this month        
        rg_otids, rg_tids, dels = [], [], []
        for t in cs:
            rgs = np.sum(t.events.loc[t.events.time >= days,"event"] == "rg")
            if rgs > 0:
                rg_tids.append(t.tid)
                rg_otids.append(int(t.events.loc[t.events.event == "rg","otid"]))
        
        for o in range(len(rg_otids)): # For each dead track
            # Note the position of the dead track object
            dels.append(np.where(sys_tids == rg_otids[o])[0][-1])
            
            # Extract the dead track object
            tDead = cs[dels[o]] # Def of regenesis requires that primary track has experienced type 3 lysis
            
            # Extract the regenerated track object
            sid_cands = np.where(sys_tids == rg_tids[o])[0] # Candidate track objects
            # Does this track have a regeneration? And does it begin when the dead track ends?
            sid_rgcode = np.array([("rg" in list(cs[sidc].events.event) ) and \
                ( cs[sidc].data.time.iloc[0] == tDead.events.loc[tDead.events.event == "ly","time"].iloc[-1] ) \
                for sidc in sid_cands])
            sid = sid_cands[np.where(sid_rgcode == 1)[0][0]] # sid of the regenerated track
            tRegen = cs[sid]
            
            # Splice together with the regenerated track
            cs[sid].data = tDead.data[:-1].append(tRegen.data,ignore_index=1, sort=1)
            cs[sid].events = tDead.events[:-1].append(tRegen.events.loc[(tRegen.events.event != "ge") | \
                (tRegen.events.time > tRegen.events.time.iloc[0])],ignore_index=1, sort=1)
            cs[sid].data.loc[cs[sid].data.Erg > 0,"Ely"] = 0
            cs[sid].data.loc[cs[sid].data.Erg > 0,"Ege"] = 0
        
        # CLEAN UP
        # Remove the dead tracks from the current month
        cs = np.delete(cs,dels)
        
        # Re-format SIDs
        for c in range(len(cs)):
            cs[c].sid = c
    
    return list(cs), list(cs0)

'''###########################
Write a Numpy Array to File Using a Gdal Object as Reference
###########################'''
# def writeNumpy_gdalObj(npArrays,outName,gdalObj,dtype=gdal.GDT_Byte):
#     '''Write a numpy array or list of arrays to a raster file using a gdal object for geographic information.
#     If a list of arrays is provided, each array will be a band in the output.
    
#     npArrays = The numpy array or list of arrays to write to disk.  All arrays must have the same dimensions.\n
#     outName = The name of the output (string)\n
#     gdalObj = An object of osgeo.gdal.Dataset class\n
#     dtype = The data type for each cell; 8-bit by default (0 to 255)
#     '''
#     # If a single array, convert to 1-element list:
#     if str(type(npArrays)) != "<type 'list'>":
#         npArrays = [npArrays]
        
#     # Convert any non-finite values to -99:
#     for i in range(len(npArrays)): # for each band...
#         npArrays[i] = np.where(np.isfinite(npArrays[i]) == 0, -99, npArrays[i])
    
#     # Create and register driver:
#     driver = gdalObj.GetDriver()
#     driver.Register()
    
#     # Create file:
#     outFile = driver.Create(outName,npArrays[0].shape[1],npArrays[0].shape[0],len(npArrays),dtype) # Create file
#     for i in range(len(npArrays)): # for each band...
#         outFile.GetRasterBand(i+1).WriteArray(npArrays[i],0,0) # Write array to file
#         outFile.GetRasterBand(i+1).ComputeStatistics(False) # Compute stats for display purposes
#     outFile.SetGeoTransform(gdalObj.GetGeoTransform()) # Set geotransform (those six needed values)
#     outFile.SetProjection(gdalObj.GetProjection())  # Set projection
    
#     outFile = None
    
#     return

'''###########################
Calculate the Mean Array of a Set of Arrays
###########################'''
def meanArrays(arrays,dtype=float):
    '''Given a list of numpy arrays of the same dimensions, this function will 
    perform raster addition on the entire set and then divide by the number of arrays.
    Returns a numpy array with the same dimensions as the inputs and a data type 
    determined by the user.
    
    arrays = A list or tuple of numpy arrays
    dtype = The desired output data type for array elements (defaults to python float)
    '''
    sums = np.zeros(arrays[0].shape, dtype=dtype)
    n = len(arrays)
    for i in range(n):
        sums = sums + arrays[i]
    mean = sums/n
    
    return mean

def meanArrays_nan(arrays,dtype=float):
    '''Given a list of numpy arrays of the same dimensions, this function will 
    perform raster addition (skipping nan values) on the entire set and then 
    divide by the number of (non-nan) arrays. Returns a numpy array with the 
    same dimensions as the inputs and a data type determined by the user.
    
    arrays = A list or tuple of numpy arrays
    dtype = The desired output data type for array elements (defaults to python float)
    '''
    sums = np.zeros(arrays[0].shape, dtype=dtype) # sum array
    n = np.zeros(arrays[0].shape, dtype=dtype) # number of non-nans array
    for i in range(len(arrays)):
        add_sums = np.where(np.isnan(arrays[i]) == 1,0,arrays[i]) # Turn nans into 0s
        add_n = np.where(np.isnan(arrays[i]) == 1,0,1) # Turns nans into 0s, all else into 1s
        sums = sums + add_sums # Add away!
        n = n + add_n # Add away!
    
    mean = sums/n # Calculate mean on gridcell-by-gridcell basis.
    
    return mean

'''###########################
Calculate the Sum Array of a Set of Arrays
###########################'''
def addArrays_nan(arrays,dtype=float):
    '''Given a list of numpy arrays of the same dimensions, this function will 
    perform raster addition (skipping nan values) on the entire set and then 
    divide by the number of (non-nan) arrays.Returns a numpy array with the 
    same dimensions as the inputs and a data type determined by the user.
    
    arrays = A list or tuple of numpy arrays
    dtype = The desired output data type for array elements (defaults to python float)
    '''
    sums = np.zeros(arrays[0].shape, dtype=dtype) # sum array
    nans = np.zeros(arrays[0].shape, dtype=dtype)
    n = len(arrays)
    for i in range(n):
        add_sums = np.where(np.isnan(arrays[i]) == 1,0,arrays[i]) # Turn nans into 0s
        sums = sums + add_sums # Add away!
        
        add_nans = np.where(np.isnan(arrays[i]) == 1,1,0) # Turns nans into 1s, all else into 0s
        nans = nans + add_nans # The higher the number, the more nans that location saw
    
    # Quality Control -- if all values were nans, make it a nan, not 0!
    nans = np.where(nans == n,np.nan,0)
    sums = sums + nans
    
    return sums 

'''###########################
Calculate the Circular Mean Array of a Set of Arrays
###########################'''
def meanCircular(a,amin,amax,favorMax=1):
    '''Given a list of values (a), this function appends one member at a time. 
    Weights are assigned for each appending step based on how many members 
    have already been appended. If the resultant mean crosses the min/max 
    boundary at any point, it is redefined as is the min and max are the same 
    point.
    
    a = a list of values
    amin = the minimum on the circle
    amax = the maximum on the circle
    favorMax = if 1 (default), the maximum is always returned (never the minimum);
        if 0, the minimum is always returned (never the maximum)
    '''
    circum = amax-amin # calculate range (circumference) of the circle
    # Sort from largest to smallest
    aa = copy.deepcopy(a)
    aa.sort()
    aa.reverse()
    
    for i,v in enumerate(aa): # for each value in a
        v = float(v)
        if i == 0: # if it's the first value, value = mean
            mean = v
        else: # otherwise...
            arc = mean-v # calculate arc length that doesn't cross min/max
            if arc/circum < 0.5: # if that arc length is less than half the total circumference
                mean = mean*i/(i+1) + v/(i+1) # then weight everything like normal; making the mean smaller
            else: # otherwise, the influence will pull the mean toward min/max
                mean = mean*i/(i+1) + (v+circum)/(i+1) # making the mean larger
            
            # After nudging the mean, check to see if it crossed the min/max line
            ## Adjust value if necessary
            if mean < amin:
                mean = amax-amin+mean
            elif mean > amax:
                mean = amin-amax+mean
    
    # Check for minimum/maximum and replace if necessary
    if mean == amin and favorMax == 1:
        mean = amax
    
    elif mean == amax and favorMax == 0:
        mean = amin
    
    return mean

def meanArraysCircular_nan(arrays,amin,amax,favorMax=1,dtype=float):
    '''Given a list of arrays, this function applies a circular mean function
    to each array location across all arrays. Returns an array with the same
    dimensions as the inputs (all of which must have the same dimensions).
    NaNs are eliminated from consideration.
    
    a = a list of values
    amin = the minimum on the circle
    amax = the maximum on the circle
    favorMax = if 1 (default), the maximum is always returned (never the minimum);
        if 0, the minimum is walawyas returned (never the maximum)
    arrays = A list or tuple of numpy arrays
    dtype = The desired output data type for array elements (defaults to python float)
    '''
    means = np.zeros(arrays[0].shape, dtype=dtype)
    
    # Loop through each location
    for r in range(arrays[0].shape[0]):
        for c in range(arrays[0].shape[1]):
            # Collect values across all arrays
            #print(str(r) + " of " + str(arrays[0].shape[0]) + ", " + str(c) + " of " + str(arrays[0].shape[1]))
            a = []
            for i in range(len(arrays)):
                if np.isnan(arrays[i][r,c]) == 0: # Only if a non-nan value
                    a.append(arrays[i][r,c])
                
                # Calculate circular mean
                if len(a) > 0:
                    means[r,c] = meanCircular(a,amin,amax,favorMax=favorMax)
                else:
                    means[r,c] = np.nan
    
    return means

'''###########################
Aggregate the Event Frequency for a Month of Cyclone Tracks
###########################'''
def aggregateEvents(tracks,typ,days,shape):
    '''Aggregates cyclone events (geneis, lysis, splitting, and merging) for 
    a given month and returns a list of 4 numpy arrays in the order
    [gen, lys, spl, mrg] recording the event frequency
    
    tracks = a list or tuple of tracks in the order [trs,trs0,trs2], where 
    trs = the current month, trs0 = the previous month, and trs2 = the active
        tracks remaining at the end of the current month
    typ = "cyclone" or "system"
    days = the time in days since a common reference date for 0000 UTC on the 
        1st day of the current month
    shape = a tuple of (r,c) where r and c are the number of rows and columns,
        respectively, for the output
    '''
    fields = [np.zeros(shape),np.zeros(shape),np.zeros(shape),np.zeros(shape)]
    
    if typ.lower() == "cyclone":
        excludeType = 2
    else:
        excludeType = 1
    
    # Limit events to only those tracks that satisfy above criteria
    tids = [tr.tid for tr in tracks[0]]
    ftids = [tr.ftid for tr in tracks[0]]
    tids0 = [tr.tid for tr in tracks[1]]
    ftids2 = [tr.ftid for tr in tracks[2]]
    
    for tr in tracks[0]: # For each track
        # Record first and last instance as genesis and lysis, respectively
        fields[0][int(tr.data.y.iloc[0]),int(tr.data.x.iloc[0])] =  fields[0][int(tr.data.y.iloc[0]),int(tr.data.x.iloc[0])] + 1
        fields[1][int(tr.data.y.iloc[-1]),int(tr.data.x.iloc[-1])] =  fields[1][int(tr.data.y.iloc[-1]),int(tr.data.x.iloc[-1])] + 1
        
        for e in range(len(tr.events)): # Check the stats for each event
            if tr.events.Etype.iloc[e] != excludeType: # Area-only or Point-only events may not be of interest
                y = int( tr.events.y.iloc[e] )
                x = int( tr.events.x.iloc[e] )
                # For splits, merges, and re-genesis, only record the event if the
                ## interacting track also satisfies the lifespan/track length criteria
                # If the event time occurs during the month of interest...
                # Check if the otid track exists in either this month or the next month:
                if tr.events.time.iloc[e] >= days and ( (tr.events.otid.iloc[e] in tids) or (tr.events.otid.iloc[e] in ftids2) ):
                    # And if so, record the event type
                    if tr.events.event.iloc[e] == "mg":
                        fields[3][y,x] =  fields[3][y,x] + 1
                    elif tr.events.event.iloc[e] == "sp":
                        fields[2][y,x] =  fields[2][y,x] + 1
                # If the event time occurs during the previous month...
                # Check if the otid track exists in either this month or the previous month:
                elif tr.events.time.iloc[e] < days and ( (tr.events.otid.iloc[e] in tids0) or (tr.events.otid.iloc[e] in ftids) ):
                    # And if so, record the event type
                    if tr.events.event.iloc[e] == "mg":
                        fields[3][y,x] =  fields[3][y,x] + 1
                    elif tr.events.event.iloc[e]== "sp":
                        fields[2][y,x] =  fields[2][y,x] + 1
    
    return fields

'''###########################
Aggregate Track-wise Stats for a Month of Cyclone Tracks
###########################'''
def aggregateTrackWiseStats(trs,date,shape):
    '''Aggregates cyclone stats that have a single value for each track:
    max propagation speed, max deepening rate, max depth, min central pressure, 
    max laplacian of central pressure. Returns a list containing five numpy 
    arrays for the frequency of the extremes at  each location in the order: 
    max propagation speed, max deepening rate, 
    max depth, min central pressure, max laplacian of central pressure
    
    trs = List of cyclone track objects for current month
    date = A date in the format [Y,M,D] or [Y,M,D,H,M,S]
    shape = a tuple of (r,c) where r and c are the number of rows and columns,
        respectively, for the output
    '''
    # Prep inputs
    maxuv_field, maxdpdt_field, maxdep_field, minp_field, maxdsqp_field = \
    np.zeros(shape),np.zeros(shape),np.zeros(shape),np.zeros(shape),np.zeros(shape)
    
    # Look at each track and aggregate stats
    for tr in trs:
        # Collect Track-Wise Stats
        trmaxuv = tr.maxUV()
        for i in range(len(trmaxuv[1])):
            maxuv_field[trmaxuv[2][i],trmaxuv[3][i]] = maxuv_field[trmaxuv[2][i],trmaxuv[3][i]] + 1
        trmaxdpdt = tr.maxDpDt()
        for i in range(len(trmaxdpdt[1])):
            maxdpdt_field[trmaxdpdt[2][i],trmaxdpdt[3][i]] = maxdpdt_field[trmaxdpdt[2][i],trmaxdpdt[3][i]] + 1
        trmaxdsqp = tr.maxDsqP()
        for i in range(len(trmaxdsqp[1])):
            maxdsqp_field[trmaxdsqp[2][i],trmaxdsqp[3][i]] = maxdsqp_field[trmaxdsqp[2][i],trmaxdsqp[3][i]] + 1
        trmaxdep = tr.maxDepth()
        for i in range(len(trmaxdep[1])):
            maxdep_field[trmaxdep[2][i],trmaxdep[3][i]] = maxdep_field[trmaxdep[2][i],trmaxdep[3][i]] + 1
        trminp = tr.minP()
        for i in range(len(trminp[1])):
            minp_field[trminp[2][i],trminp[3][i]] = minp_field[trminp[2][i],trminp[3][i]] + 1
    
    return [maxuv_field, maxdpdt_field, maxdep_field, minp_field, maxdsqp_field]

'''###########################
Aggregate Point-wise Stats for a Month of Cyclone Tracks
###########################'''
def aggregatePointWiseStats(trs,n,shape):
    '''Aggregates cyclone counts, track density, and a host of other Eulerian
    measures of cyclone characteristics. Returns a list of numpy arrays in the
    following order: track density, cyclone center frequnecy, cyclone center 
    frequency for centers with valid pressure, and multi-center cyclone 
    frequnecy; the average propagation speed, propogation direction, radius, 
    area, depth, depth/radius, deepening rate, central pressure, and laplacian 
    of central pressure.
     
    trs = List of cyclone track objects for current month
    n = The number of time slices considered in the creation of trs (usually the 
        number of days in the given month times 24 hours divided by the time interval in hours)
    shape = A tuple of (r,c) where r and c are the number of rows and columns,
        respectively, for the output
    '''
    # Ensure that n is a float
    n = float(n)
    
    # Create empty fields
    sys_field, trk_field, countU_field, countA_field, countP_field = \
    np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)
    
    pcent_field, dpdt_field, dpdr_field, dsqp_field, depth_field = \
    np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)
    
    uvDi_fields, uvAb_field, radius_field, area_field, mcc_field = \
    [], np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)
    
    for tr in trs:
        uvDi_field = np.zeros(shape)
        
        # Count Point-Wise Stats
        trk_tracker = np.zeros(shape) # This array tracks whether the track has been counted yet in each grid cell
        for i in range(len(tr.data))[:-1]:
            x = int(tr.data.x.iloc[i])
            y = int(tr.data.y.iloc[i])
            
            # Existance of System/Track
            sys_field[y,x] = sys_field[y,x] + 1
            if trk_tracker[y,x] == 0: # Only count in trk_field if it hasn't yet been counted there!
                trk_field[y,x] = trk_field[y,x] + 1
                trk_tracker[y,x] = trk_tracker[y,x] + 1
            # Special Cases:
            if i > 0:
                countU_field[y,x] = countU_field[y,x] + 1
            if tr.data.radius.iloc[i] != 0:
                countA_field[y,x] = countA_field[y,x] + 1
            if np.isnan(tr.data.p_cent.iloc[i]) != 1:
                countP_field[y,x] = countP_field[y,x] + 1
            
            # Other Eulerian Measures
            pcent_field[y,x] = pcent_field[y,x] + float(np.where(np.isnan(tr.data.p_cent.iloc[i]) == 1,0,tr.data.p_cent.iloc[i]))
            dpdt_field[y,x] = dpdt_field[y,x] + float(np.where(np.isnan(tr.data.DpDt.iloc[i]) == 1,0,tr.data.DpDt.iloc[i]))
            dpdr_field[y,x] = dpdr_field[y,x] + float(np.where(np.isnan(tr.data.DpDr.iloc[i]) == 1,0,tr.data.DpDr.iloc[i]))
            dsqp_field[y,x] = dsqp_field[y,x] + float(np.where(np.isnan(tr.data.DsqP.iloc[i]) == 1,0,tr.data.DsqP.iloc[i]))
            depth_field[y,x] = depth_field[y,x] + float(np.where(np.isnan(tr.data.depth.iloc[i]) == 1,0,tr.data.depth.iloc[i]))
            uvAb_field[y,x] = uvAb_field[y,x] + float(np.where(np.isnan(tr.data.uv.iloc[i]) == 1,0,tr.data.uv.iloc[i]))
            uvDi_field[y,x] = uvDi_field[y,x] + vectorDirectionFrom(tr.data.u.iloc[i],tr.data.v.iloc[i])
            radius_field[y,x] = radius_field[y,x] + float(np.where(np.isnan(tr.data.radius.iloc[i]) == 1,0,tr.data.radius.iloc[i]))
            area_field[y,x] = area_field[y,x] + float(np.where(np.isnan(tr.data.area.iloc[i]) == 1,0,tr.data.area.iloc[i]))
            mcc_field[y,x] = mcc_field[y,x] + float(np.where(float(tr.data.centers.iloc[i]) > 1,1,0))
        
        uvDi_fields.append(np.where(uvDi_field == 0,np.nan,uvDi_field))
        
    ### AVERAGES AND DENSITIES ###
    uvDi_fieldAvg = meanArraysCircular_nan(uvDi_fields,0,360)
    
    pcent_fieldAvg = pcent_field/countP_field
    dpdt_fieldAvg = dpdt_field/countU_field
    dpdr_fieldAvg = dpdr_field/countA_field
    dsqp_fieldAvg = dsqp_field/countP_field
    depth_fieldAvg = depth_field/countA_field
    uvAb_fieldAvg = uvAb_field/countU_field
    radius_fieldAvg = radius_field/countP_field
    area_fieldAvg = area_field/countP_field
    
    return [trk_field, sys_field/n, countP_field/n, countU_field/n, countA_field/n, mcc_field/n, \
        uvAb_fieldAvg, uvDi_fieldAvg, radius_fieldAvg, area_fieldAvg, depth_fieldAvg, \
        dpdr_fieldAvg, dpdt_fieldAvg, pcent_fieldAvg, dsqp_fieldAvg]

'''###########################
Aggregate Fields that Exist for Each Time Step in a Month of Cyclone Tracking
###########################'''
def aggregateTimeStepFields(inpath,trs,mt,timestep,dateref=[1900,1,1,],lyb=1,dpy=365):
    '''Aggregates fields that exist for each time step in a month of cyclone
    tracking data. Returns a list of numpy arrays.
    
    inpath = a path to the directory for the cyclone detection/tracking output
    (should end with the folder containing an "AreaFields" folder)
    mt = month time in format [Y,M,1] or [Y,M,1,0,0,0]
    timestep = timestep in format [Y,M,D] or [Y,M,D,H,M,S]
    lys = 1 for Gregorian calendar, 0 for 365-day calendar
    '''
    # Supports
    monthstep = [0,1,0,0,0,0]
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    mons = ["01","02","03","04","05","06","07","08","09","10","11","12"]
    days = ["01","02","03","04","05","06","07","08","09","10","11","12","13",\
        "14","15","16","17","18","19","20","21","22","23","24","25","26","27",\
        "28","29","30","31"]
    hours = ["0000","0100","0200","0300","0400","0500","0600","0700","0800",\
        "0900","1000","1100","1200","1300","1400","1500","1600","1700","1800",\
        "1900","2000","2100","2200","2300"]
    
    # Start timers
    t = mt
    tcount = 0
    
    # Create an empty array to start
    date = str(t[0])+mons[t[1]-1]+days[t[2]-1]+"_"+hours[t[3]]
    cf = pd.read_pickle(inpath[:-7]+"/CycloneFields/"+str(t[0])+"/"+months[t[1]-1]+"/CF"+date+".pkl")
    fieldAreas = 0*cf.fieldAreas
    
    while t != timeAdd(mt,monthstep):
        date = str(t[0])+mons[t[1]-1]+days[t[2]-1]+"_"+hours[t[3]]
        
        # Load Cyclone Field for this time step
        cf = pd.read_pickle(inpath[:-7]+"/CycloneFields/"+str(t[0])+"/"+months[t[1]-1]+"/CF"+date+".pkl")
        cAreas, nC = ndimage.measurements.label(cf.fieldAreas)
        
        # For each track...
        for tr in trs:
            d = daysBetweenDates(dateref,t,lyb,dpy)
            try:
                x = int(tr.data.loc[(tr.data.time == d) & (tr.data.type != 0),"x"].iloc[0])
                y = int(tr.data.loc[(tr.data.time == d) & (tr.data.type != 0),"y"].iloc[0])
                
                # Add the area for this time step
                fieldAreas = fieldAreas + np.where(cAreas == cAreas[y,x], 1, 0)
            
            except:
                continue
        
        # Increment time step
        tcount = tcount+1
        t = timeAdd(t,timestep,lyb,dpy)
    return [fieldAreas/tcount]

'''###########################
Calculate the Distance Between Two (Lat,Long) Locations
###########################'''
def haversine(lats1, lats2, longs1, longs2, units="meters"):
    '''This function uses the haversine formula to calculate the distance between
    two points on the Earth's surface when given the latitude and longitude (in decimal 
    degrees). It returns a distance in the units specified (default is meters). If 
    concerned with motion, (lats1,longs1) is the initial position and (lats2,longs2)
    is the final position.
    '''
    import numpy as np
    
    # Convert to radians:
    lat1, lat2 = lats1*np.pi/180, lats2*np.pi/180
    long1, long2 = longs1*np.pi/180, longs2*np.pi/180
    
    # Perform distance calculation:
    R = 6371000
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((long2-long1)/2)**2
    c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
    d = R*c
    
    # Conversions:
    if units.lower() in ["m","meters","metres","meter","metre"]:
        d = d
    elif units.lower() in ["km","kms","kilometer","kilometre","kilometers","kilometres"]:
        d = d/1000
    elif units.lower() in ["ft","feet"]:
        d = d*3.28084
    elif units.lower() in ["mi","miles"]:
        d = d*0.000621371
    
    return d

'''###########################
Calculate the Direction a Vector is Coming From
###########################'''
def vectorDirectionFrom(u,v,deg=1):
    '''This function calculates the direction a vector is coming from when given
    a u and v component. By default, it will return a value in degrees. Set
    deg = 0 to get radians instead.
    
    Returns a value in the range (0,360], with a 0 indicating no movement.
    '''
    # Take the 180 degree arctangent
    ### Rotate results counter-clockwise by 90 degrees
    uvDi = 0.5*np.pi - np.arctan2(-v,-u) # Use negatives because it's FROM
    
    # If you get a negative value (or 0), add 2 pi to make it positive
    if uvDi <= 0:
        uvDi = uvDi+2*np.pi
    
    # Set to 0 if no motion occurred
    if u == 0 and v == 0:
        uvDi = 0
    
    # If the answer needs to be in degrees, convert
    if deg == 1:
        uvDi = 180*uvDi/np.pi
    
    return uvDi

'''###########################
Create a Discrete Array from a Continuous Array
###########################'''
def toDiscreteArray(inArray, breaks):
    '''
    Given a continuous numpy array, converts all values into discrete (ordinal)
    bins. Values that do not fall within the defined breaks are reclassified
    as np.nan.\n
    
    inArray = A numpy array of continuous numerical data
    breaks = A list of *fully inclusive* breakpoints. Use -np.inf and np.inf
    for extending the color bar beyond a minimum and maximum, respectively.
    All bins are b[i-1] <= x < b[i] except for the final bin, which is
    b[n-2] <= x <= b[n-1], where n is the number of breaks [n = len(breaks)].
    
    '''
    # Initialize output as all NaN values
    outArray = np.zeros_like(inArray)*np.nan
    
    # Set the start and end of the while loop
    b = 1
    end = len(breaks) - 1
    
    # Discretize each bin that has an <= x < structure
    while b < end:
        outArray[np.where((inArray >= breaks[b-1]) & (inArray < breaks[b]))] = b
        b += 1
    
    # Discretize the final bin, which has an <= x <= structure
    outArray[np.where((inArray >= breaks[b-1]) & (inArray <= breaks[b]))] = b
    
    return outArray

'''##########################
Linear Model (OLS Regression) for 3-D array with shape (t,y,x)
##########################'''
def lm(x,y,minN=10):
    '''Calculates a ordinary least squares regression model. Designed
    specifically for quick batch trend analayses.  Does not consider any
    autoregression.\n
    
    x = the time variable (a list or 1-D numpy array)\n
    y = the dependent variable (a 3-D numpy array where axis 0 is the time axis)\n
    minN = minimum number of non-NaN values required for y in the x dimension 
    (e.g., number of years with valid data)\n
    returns 5 numpy arrays with the same dimensions as axis 1 and axis 2 of y:
    b (the slope coefficient), a (the intercept coefficient), r^2, 
    p-value for b, and standard error for b
    '''
    # Create empty arrays for output
    b, a, r, p, se = np.zeros_like(y[0])*np.nan, np.zeros_like(y[0])*np.nan, np.zeros_like(y[0])*np.nan, np.zeros_like(y[0])*np.nan, np.zeros_like(y[0])*np.nan
    
    # Find locations with at least minN finite values
    n = np.isfinite(y).sum(axis=0)
    validrows, validcols = np.where( n >= minN )
    
    # For each row/col
    for i in range(validrows.shape[0]):
        ro, co = validrows[i], validcols[i]
        
        yin = y[:,ro,co]
        
        # Create a linear model
        b[ro,co], a[ro,co], r[ro,co], p[ro,co], se[ro,co] =  stats.linregress(x[np.isfinite(yin)],yin[np.isfinite(yin)])
    
    return b, a, r*r, p, se

'''#########################
Compare Tracks from Different Datasets
#########################'''
def comparetracks(trs1,trs2,trs2b,date1,refdate=[1900,1,1,0,0,0],minmatch=0.6,maxsep=500,system=True,lyb=1,dpy=365):
    '''This function performs a track-matching comparison between two different
    sets of tracks. The tracks being compared should be from the same month and 
    have the same temporal resolution. They should differ based on input data,
    spatial resolution, or detection/tracking parameters. The function returns
    a pandas dataframe with a row for each cyclone track in first dataset. If 
    one exists, the cyclone track in the second dataset that best matches is 
    compared by the separation distance and intensity differences (central
    pressure, its Laplacian, area, and depth).\n
    
    trs1 = A list of cyclone track objects from the first version
    trs2 = A list of cyclone track objects from the second version; must be for
        the same month as trs1
    trs2b = A list of cylone track objects from the second vesion; must be for
        one month prior to trs1
    date1 = A date in the format [YYYY,MM,1,0,0,0], corresponds to trs1
    refdate = The reference date used during cyclone tracking in the format
        [YYYY,MM,DD,HH,MM,SS], by default [1900,1,1,0,0,0]
    minmatch = The minimum ratio of matched times to total times for any two
        cyclone tracks to be considered a matching pair... using the equation
        2*N(A & B) / (N(A) + N(B)) >= minmatch, where N is the number of 
        observations times, and A and B are the tracks being compared
    maxsep = The maximum allowed separation between a matching pair of tracks;
        separation is calculated as the average distance between the tracks 
        during matching observation times using the Haversine formula
    system = whether comparison is between system tracks or cyclone tracks;
        default is True, meaning that system tracks are being compared.
    '''
    refday = daysBetweenDates(refdate,date1,lyb,dpy)
    timeb = timeAdd(date1,[0,-1,0],lyb,dpy)
    
    ##### System Tracks #####
    if system == True:
        pdf = pd.DataFrame()
             
        # For each track in version 1, find all of the version 2 tracks that overlap at least *minmatch* (e.g. 60%) of the obs times
        for i1 in range(len(trs1)):
            # Extract the observation times for the version 1 track
            times1 = np.array(trs1[i1].data.time[trs1[i1].data.type != 0])
            lats1 = np.array(trs1[i1].data.lat[trs1[i1].data.type != 0])
            lons1 = np.array(trs1[i1].data.long[trs1[i1].data.type != 0])
            
            ids2, ids2b = [], [] # Lists in which to store possible matches from version 2
            avgdist2, avgdist2b = [], [] # Lits in which to store the average distances between cyclone tracks
            for i2 in range(len(trs2)):
                # Extract the observation times for the version 2 track
                times2 = np.array(trs2[i2].data.time[trs2[i2].data.type != 0])
                # Assess the fraction of matching observations
                matchfrac = 2*np.sum([t in times2 for t in times1]) / float(len(times1) + len(times2))
                # If that's satisfied, calculate the mean separation for matching observation times
                if matchfrac >= minmatch:
                    timesm = [t for t in times1 if t in times2] # Extract matched times
                    lats2 = np.array(trs2[i2].data.lat[trs2[i2].data.type != 0])
                    lons2 = np.array(trs2[i2].data.long[trs2[i2].data.type != 0])
                    
                    # Calculate the mean separation between tracks
                    avgdist2.append( np.mean( [haversine(lats1[np.where(times1 == tm)][0],lats2[np.where(times2 == tm)][0],\
                        lons1[np.where(times1 == tm)][0],lons2[np.where(times2 == tm)][0],units='km') for tm in timesm] ) )
                    
                    # And store the track id for the version 2 cyclone
                    ids2.append(i2)
            
            # If the version 1 track also existed last month, check last month's version 2 tracks, too... 
            if times1[0] < refday:
                for i2b in range(len(trs2b)):
                    # Extract the observation times for the version 2b track
                    times2b = np.array(trs2b[i2b].data.time[trs2b[i2b].data.type != 0])
                    # Assess the fraction of matching observations
                    matchfrac = 2*np.sum([t in times2b for t in times1]) / float(len(times1) + len(times2b))
                    if matchfrac >= minmatch:
                        timesmb = [t for t in times1 if t in times2b] # Extract matched times
                        lats2b = np.array(trs2b[i2b].data.lat[trs2b[i2b].data.type != 0])
                        lons2b = np.array(trs2b[i2b].data.long[trs2b[i2b].data.type != 0])
                        
                        # Calculate the mean separation between tracks
                        avgdist2b.append( np.mean( [haversine(lats1[np.where(times1 == tmb)][0],lats2b[np.where(times2b == tmb)][0],\
                            lons1[np.where(times1 == tmb)][0],lons2b[np.where(times2b == tmb)][0],units='km') for tmb in timesmb] ) )
                        
                        # And store the track id for the version 2b cyclone
                        ids2b.append(i2b)
                
                # Identify how many possible matches are satisfy the maximum average separation distance
                nummatch = np.where(np.array(avgdist2+avgdist2b) < maxsep)[0].shape[0]
                
                # Determine which version 2(b) track has the shortest average separation
                if nummatch == 0: # If there's no match...
                    pdf = pdf.append(pd.DataFrame([{"Year1":date1[0],"Month1":date1[1],"sid1":trs1[i1].sid,"Num_Matches":nummatch,"Year2":np.nan,"Month2":np.nan,\
                        "sid2":np.nan,"Dist":np.nan,"pcentDiff":np.nan,"areaDiff":np.nan,"depthDiff":np.nan,"dsqpDiff":np.nan},]), ignore_index=1, sort=1)
                
                elif np.min(avgdist2+[np.inf]) > np.min(avgdist2b+[np.inf]): # If the best match is from previous month...
                    im = ids2b[np.where(avgdist2b == np.min(avgdist2b))[0][0]]
                    timesmb = [t for t in times1 if t in np.array(trs2b[im].data.time[trs2b[im].data.type != 0])] # Extract matched times
                    
                    # Find average intensity differences
                    areaDiff = np.mean([float(trs1[i1].data.loc[trs1[i1].data.time == tmb,"area"]) - float(trs2b[im].data.loc[trs2b[im].data.time == tmb,"area"]) for tmb in timesmb])
                    pcentDiff = np.mean([float(trs1[i1].data.loc[trs1[i1].data.time == tmb,"p_cent"]) - float(trs2b[im].data.loc[trs2b[im].data.time == tmb,"p_cent"]) for tmb in timesmb])
                    dsqpDiff = np.mean([float(trs1[i1].data.loc[trs1[i1].data.time == tmb,"DsqP"]) - float(trs2b[im].data.loc[trs2b[im].data.time == tmb,"DsqP"]) for tmb in timesmb])
                    depthDiff = np.mean([float(trs1[i1].data.loc[trs1[i1].data.time == tmb,"depth"]) - float(trs2b[im].data.loc[trs2b[im].data.time == tmb,"depth"]) for tmb in timesmb])
                    
                    pdf = pdf.append(pd.DataFrame([{"Year1":date1[0],"Month1":date1[1],"sid1":trs1[i1].sid,"Num_Matches":nummatch,"Year2":timeb[0],"Month2":timeb[1],"sid2":trs2b[im].sid,\
                        "Dist":np.min(avgdist2b),"pcentDiff":pcentDiff,"areaDiff":areaDiff,"depthDiff":depthDiff,"dsqpDiff":dsqpDiff},]), ignore_index=1, sort=1)
                
                else: # If the best match is from current month...
                    im = ids2[np.where(avgdist2 == np.min(avgdist2))[0][0]]
                    timesm = [t for t in times1 if t in np.array(trs2[im].data.time[trs2[im].data.type != 0])] # Extract matched times
                    
                    # Find average intensity differences
                    areaDiff = np.mean([float(trs1[i1].data.loc[trs1[i1].data.time == tm,"area"]) - float(trs2[im].data.loc[trs2[im].data.time == tm,"area"]) for tm in timesm])
                    pcentDiff = np.mean([float(trs1[i1].data.loc[trs1[i1].data.time == tm,"p_cent"]) - float(trs2[im].data.loc[trs2[im].data.time == tm,"p_cent"]) for tm in timesm])
                    dsqpDiff = np.mean([float(trs1[i1].data.loc[trs1[i1].data.time == tm,"DsqP"]) - float(trs2[im].data.loc[trs2[im].data.time == tm,"DsqP"]) for tm in timesm])
                    depthDiff = np.mean([float(trs1[i1].data.loc[trs1[i1].data.time == tm,"depth"]) - float(trs2[im].data.loc[trs2[im].data.time == tm,"depth"]) for tm in timesm])
                    
                    pdf = pdf.append(pd.DataFrame([{"Year1":date1[0],"Month1":date1[1],"sid1":trs1[i1].sid,"Num_Matches":nummatch,"Year2":date1[0],"Month2":date1[1],"sid2":trs2[im].sid,\
                        "Dist":np.min(avgdist2),"pcentDiff":pcentDiff,"areaDiff":areaDiff,"depthDiff":depthDiff,"dsqpDiff":dsqpDiff},]), ignore_index=1, sort=1)
            
            # If the version 1 track only existed in the current month...
            else:
                # Identify how many possible matches are satisfy the maximum average separation distance
                nummatch = np.where(np.array(avgdist2) < maxsep)[0].shape[0]
                
                # Determine which version 2 track has the shortest average separation
                if nummatch == 0: # If there's no match...
                    pdf = pdf.append(pd.DataFrame([{"Year1":date1[0],"Month1":date1[1],"sid1":trs1[i1].sid,"Num_Matches":nummatch,"Year2":np.nan,"Month2":np.nan,\
                        "sid2":np.nan,"Dist":np.nan,"pcentDiff":np.nan,"areaDiff":np.nan,"depthDiff":np.nan,"dsqpDiff":np.nan},]), ignore_index=1, sort=1)
                
                else: # If the best match is from current month...
                    im = ids2[np.where(avgdist2 == np.min(avgdist2))[0][0]]
                    timesm = [t for t in times1 if t in np.array(trs2[im].data.time[trs2[im].data.type != 0])] # Extract matched times
                    
                    # Find average intensity differences
                    areaDiff = np.mean([float(trs1[i1].data.loc[trs1[i1].data.time == tm,"area"]) - float(trs2[im].data.loc[trs2[im].data.time == tm,"area"]) for tm in timesm])
                    pcentDiff = np.mean([float(trs1[i1].data.loc[trs1[i1].data.time == tm,"p_cent"]) - float(trs2[im].data.loc[trs2[im].data.time == tm,"p_cent"]) for tm in timesm])
                    dsqpDiff = np.nanmean([float(trs1[i1].data.loc[trs1[i1].data.time == tm,"DsqP"]) - float(trs2[im].data.loc[trs2[im].data.time == tm,"DsqP"]) for tm in timesm])
                    depthDiff = np.mean([float(trs1[i1].data.loc[trs1[i1].data.time == tm,"depth"]) - float(trs2[im].data.loc[trs2[im].data.time == tm,"depth"]) for tm in timesm])
                    
                    pdf = pdf.append(pd.DataFrame([{"Year1":date1[0],"Month1":date1[1],"sid1":trs1[i1].sid,"Num_Matches":nummatch,"Year2":date1[0],"Month2":date1[1],"sid2":trs2[im].sid,\
                        "Dist":np.min(avgdist2),"pcentDiff":pcentDiff,"areaDiff":areaDiff,"depthDiff":depthDiff,"dsqpDiff":dsqpDiff},]), ignore_index=1, sort=1)
        
    ####### Cyclone Tracks #######
    else:
        pdf = pd.DataFrame()
           
        # For each track in version 1, find all of the version 2 tracks that overlap at least *minmatch* (e.g. 60%) of the obs times
        for i1 in range(len(trs1)):
            # Extract the observation times for the version 1 track
            times1 = np.array(trs1[i1].data.time[trs1[i1].data.type != 0])
            lats1 = np.array(trs1[i1].data.lat[trs1[i1].data.type != 0])
            lons1 = np.array(trs1[i1].data.long[trs1[i1].data.type != 0])
            
            ids2, ids2b = [], [] # Lists in which to store possible matches from version 2
            avgdist2, avgdist2b = [], [] # Lits in which to store the average distances between cyclone tracks
            for i2 in range(len(trs2)):
                # Extract the observation times for the version 2 track
                times2 = np.array(trs2[i2].data.time[trs2[i2].data.type != 0])
                # Assess the fraction of matching observations
                matchfrac = 2*np.sum([t in times2 for t in times1]) / float(len(times1) + len(times2))
                # If that's satisfied, calculate the mean separation for matching observation times
                if matchfrac >= minmatch:
                    timesm = [t for t in times1 if t in times2] # Extract matched times
                    lats2 = np.array(trs2[i2].data.lat[trs2[i2].data.type != 0])
                    lons2 = np.array(trs2[i2].data.long[trs2[i2].data.type != 0])
                    
                    # Calculate the mean separation between tracks
                    avgdist2.append( np.mean( [haversine(lats1[np.where(times1 == tm)][0],lats2[np.where(times2 == tm)][0],\
                        lons1[np.where(times1 == tm)][0],lons2[np.where(times2 == tm)][0],units='km') for tm in timesm] ) )
                    
                    # And store the track id for the version 2 cyclone
                    ids2.append(i2)
            
            # If the version 1 track also existed last month, check last month's version 2 tracks, too... 
            if times1[0] < refday:
                for i2b in range(len(trs2b)):
                    # Extract the observation times for the version 2b track
                    times2b = np.array(trs2b[i2b].data.time[trs2b[i2b].data.type != 0])
                    # Assess the fraction of matching observations
                    matchfrac = 2*np.sum([t in times2b for t in times1]) / float(len(times1) + len(times2b))
                    if matchfrac >= minmatch:
                        timesmb = [t for t in times1 if t in times2b] # Extract matched times
                        lats2b = np.array(trs2b[i2b].data.lat[trs2b[i2b].data.type != 0])
                        lons2b = np.array(trs2b[i2b].data.long[trs2b[i2b].data.type != 0])
                        
                        # Calculate the mean separation between tracks
                        avgdist2b.append( np.mean( [haversine(lats1[np.where(times1 == tmb)][0],lats2b[np.where(times2b == tmb)][0],\
                            lons1[np.where(times1 == tmb)][0],lons2b[np.where(times2b == tmb)][0],units='km') for tmb in timesmb] ) )
                        
                        # And store the track id for the version 2b cyclone
                        ids2b.append(i2b)
                
                # Identify how many possible matches are satisfy the maximum average separation distance
                nummatch = np.where(np.array(avgdist2+avgdist2b) < maxsep)[0].shape[0]
                
                # Determine which version 2(b) track has the shortest average separation
                if nummatch == 0: # If there's no match...
                    pdf = pdf.append(pd.DataFrame([{"Year1":date1[0],"Month1":date1[1],"tid1":trs1[i1].tid,"Num_Matches":nummatch,"Year2":np.nan,"Month2":np.nan,\
                        "tid2":np.nan,"Dist":np.nan,"pcentDiff":np.nan,"areaDiff":np.nan,"depthDiff":np.nan,"dsqpDiff":np.nan},]), ignore_index=1, sort=1)
                
                elif np.min(avgdist2+[np.inf]) > np.min(avgdist2b+[np.inf]): # If the best match is from previous month...
                    im = ids2b[np.where(avgdist2b == np.min(avgdist2b))[0][0]]
                    timesmb = [t for t in times1 if t in np.array(trs2b[im].data.time[trs2b[im].data.type != 0])] # Extract matched times
                    
                    # Find average intensity differences
                    areaDiff = np.mean([float(trs1[i1].data.loc[trs1[i1].data.time == tmb,"area"]) - float(trs2b[im].data.loc[trs2b[im].data.time == tmb,"area"]) for tmb in timesmb])
                    pcentDiff = np.mean([float(trs1[i1].data.loc[trs1[i1].data.time == tmb,"p_cent"]) - float(trs2b[im].data.loc[trs2b[im].data.time == tmb,"p_cent"]) for tmb in timesmb])
                    dsqpDiff = np.mean([float(trs1[i1].data.loc[trs1[i1].data.time == tmb,"DsqP"]) - float(trs2b[im].data.loc[trs2b[im].data.time == tmb,"DsqP"]) for tmb in timesmb])
                    depthDiff = np.mean([float(trs1[i1].data.loc[trs1[i1].data.time == tmb,"depth"]) - float(trs2b[im].data.loc[trs2b[im].data.time == tmb,"depth"]) for tmb in timesmb])
                    
                    pdf = pdf.append(pd.DataFrame([{"Year1":date1[0],"Month1":date1[1],"tid1":trs1[i1].tid,"Num_Matches":nummatch,"Year2":timeb[0],"Month2":timeb[1],"tid2":trs2b[im].tid,\
                        "Dist":np.min(avgdist2b),"pcentDiff":pcentDiff,"areaDiff":areaDiff,"depthDiff":depthDiff,"dsqpDiff":dsqpDiff},]), ignore_index=1, sort=1)
                
                else: # If the best match is from current month...
                    im = ids2[np.where(avgdist2 == np.min(avgdist2))[0][0]]
                    timesm = [t for t in times1 if t in np.array(trs2[im].data.time[trs2[im].data.type != 0])] # Extract matched times
                    
                    # Find average intensity differences
                    areaDiff = np.mean([float(trs1[i1].data.loc[trs1[i1].data.time == tm,"area"]) - float(trs2[im].data.loc[trs2[im].data.time == tm,"area"]) for tm in timesm])
                    pcentDiff = np.mean([float(trs1[i1].data.loc[trs1[i1].data.time == tm,"p_cent"]) - float(trs2[im].data.loc[trs2[im].data.time == tm,"p_cent"]) for tm in timesm])
                    dsqpDiff = np.mean([float(trs1[i1].data.loc[trs1[i1].data.time == tm,"DsqP"]) - float(trs2[im].data.loc[trs2[im].data.time == tm,"DsqP"]) for tm in timesm])
                    depthDiff = np.mean([float(trs1[i1].data.loc[trs1[i1].data.time == tm,"depth"]) - float(trs2[im].data.loc[trs2[im].data.time == tm,"depth"]) for tm in timesm])
                    
                    pdf = pdf.append(pd.DataFrame([{"Year1":date1[0],"Month1":date1[1],"tid1":trs1[i1].tid,"Num_Matches":nummatch,"Year2":date1[0],"Month2":date1[1],"tid2":trs2[im].tid,\
                        "Dist":np.min(avgdist2),"pcentDiff":pcentDiff,"areaDiff":areaDiff,"depthDiff":depthDiff,"dsqpDiff":dsqpDiff},]), ignore_index=1, sort=1)
            
            # If the version 1 track only existed in the current month...
            else:
                # Identify how many possible matches are satisfy the maximum average separation distance
                nummatch = np.where(np.array(avgdist2) < maxsep)[0].shape[0]
                
                # Determine which version 2 track has the shortest average separation
                if nummatch == 0: # If there's no match...
                    pdf = pdf.append(pd.DataFrame([{"Year1":date1[0],"Month1":date1[1],"tid1":trs1[i1].tid,"Num_Matches":nummatch,"Year2":np.nan,"Month2":np.nan,\
                        "tid2":np.nan,"Dist":np.nan,"pcentDiff":np.nan,"areaDiff":np.nan,"depthDiff":np.nan,"dsqpDiff":np.nan},]), ignore_index=1, sort=1)
                
                else: # If the best match is from current month...
                    im = ids2[np.where(avgdist2 == np.min(avgdist2))[0][0]]
                    timesm = [t for t in times1 if t in np.array(trs2[im].data.time[trs2[im].data.type != 0])] # Extract matched times
                    
                    # Find average intensity differences
                    areaDiff = np.mean([float(trs1[i1].data.loc[trs1[i1].data.time == tm,"area"]) - float(trs2[im].data.loc[trs2[im].data.time == tm,"area"]) for tm in timesm])
                    pcentDiff = np.mean([float(trs1[i1].data.loc[trs1[i1].data.time == tm,"p_cent"]) - float(trs2[im].data.loc[trs2[im].data.time == tm,"p_cent"]) for tm in timesm])
                    dsqpDiff = np.nanmean([float(trs1[i1].data.loc[trs1[i1].data.time == tm,"DsqP"]) - float(trs2[im].data.loc[trs2[im].data.time == tm,"DsqP"]) for tm in timesm])
                    depthDiff = np.mean([float(trs1[i1].data.loc[trs1[i1].data.time == tm,"depth"]) - float(trs2[im].data.loc[trs2[im].data.time == tm,"depth"]) for tm in timesm])
                    
                    pdf = pdf.append(pd.DataFrame([{"Year1":date1[0],"Month1":date1[1],"tid1":trs1[i1].tid,"Num_Matches":nummatch,"Year2":date1[0],"Month2":date1[1],"tid2":trs2[im].tid,\
                        "Dist":np.min(avgdist2),"pcentDiff":pcentDiff,"areaDiff":areaDiff,"depthDiff":depthDiff,"dsqpDiff":dsqpDiff},]), ignore_index=1, sort=1)
    return pdf
