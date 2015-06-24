import numpy
import scipy
import h5py
import json
from pylab import *
from matplotlib import *
from  NearshoreWaveFunctions_3p0 import*

print('Loading HDF5 files...')

f = h5py.File('transect_data.h5') # Open the HDF5 file

# Average spatial resolution of all transects
transect_spacing = f['habitat_types'].attrs['transect_spacing']
# Distance between transect samples
model_resolution = f['habitat_types'].attrs['model_resolution']

print('average space between transects:', transect_spacing, 'm')
print('model resolution:', model_resolution, 'm')

# ------------------------------------------------
# Define file contents
# ------------------------------------------------
# Values on the transects are that of the closest shapefile point feature

#--Climatic forcing
# 5 Fields: Surge, WindSpeed, WavePeriod, WaveHeight
# Matrix format: transect_count x 5 x max_transect_length  
climatic_forcing_dataset = f['climatic_forcing']

#--Soil type
#mud=0, sand=1, gravel=2, unknown=-1
# Matrix size: transect_count x max_transect_length
soil_types_dataset = f['soil_types']

#--Soil properties:
# 0-mud: DryDensty, ErosionCst
# 1-sand: SedSize, DuneHeight, BermHeight, BermLength, ForshrSlop
# 2-gravel: same as sand
# Matrix size: transect_count x 5 x max_transect_length
soil_properties_dataset = f['soil_properties']

#--Habitat types:  !! This is different from the sheet and the content of the file. 2 is seagrass, not reef!!
#   <0 = no habitats
#   0 = kelp #   1 = eelgrass #   2 = underwater structure/oyster reef
#   3 = coral reef #   4 = levee #   5 = beach #   6 = seawall
#   7 = marsh #   8 = mangrove #   9 = terrestrial structure
# Matrix size: transect_count x max_transect_length
habitat_types_dataset = f['habitat_types']

#--Habitat properties for each habitat type:
#   <0: no data
#   0: StemHeight=0, StemDiam=1, StemDensty=2, StemDrag=3, Type=4
#   1: same as 1
#   2: ShoreDist=2, Height=3, BaseWidth=4, CrestWidth=5
#   3: FricCoverd=1, FricUncov=2, SLRKeepUp=3, DegrUncov=4
#   4: Type=1, Height=2, SideSlope=3, OvertopLim=4
#   5: SedSize=1, ForshrSlop=2, BermLength=3, BermHeight=4, DuneHeight=5
#   6: same as 5
#   7: SurfRough=1, SurRedFact=2, StemHeight=3, StemDiam=4, StemDensty=5, StemDrag=6
#   8: a lot...
#   9: Type=1, TransParam=2, Height=3habitat_properties_dataset = f['habitat_properties']
habitat_properties_dataset = f['habitat_properties']

#--Bathymetry in meters
# Matrix size: transect_count x max_transect_length
bathymetry_dataset = f['bathymetry']
# row, column (I, J resp.) position of each transect point in the raster matrix when extracted as a numpy array
# Matrix size: transect_count x 2 (1st index is I, 2nd is J) x max_transect_length

#--Index of the datapoint that corresponds to the shore pixel
positions_dataset = f['ij_positions']
# Matrix size: transect_count x max_transect_length

# Name of the "subdirectory" that contains the indices and coordinate limits described below
shore_dataset = f['shore_index']

# First and last indices of the valid transect points (that are not nodata)
limit_group = f['limits']
# First index should be 0, and second is the last index before a nodata point.
# Matrix size: transect_count x 2 (start, end) x max_transect_length
indices_limit_dataset = limit_group['indices']

#--Coordinates of the first transect point (index 0, start point) and the last valid transect point (end point)
coordinates_limits_dataset = limit_group['ij_coordinates']

# ------------------------------------------------
# Extract content
# ------------------------------------------------

#-- Climatic forcing
#transect_count, forcing, transect_length = climatic_forcing_dataset.shape
Ho=8;To=12;Uo=0;Surge=0; #Default inputs for now

# 5 Fields: Surge, WindSpeed, WavePeriod, WaveHeight
# Matrix format: transect_count x 5 x max_transect_length

#   transect_count: number of transects
#   max_transect_length: maximum possible length of a transect in pixels
#   max_habitat_field_count: maximum number of fields for habitats
#   max_soil_field_count: maximum number of fields for soil types
transect_count, habitat_fields, transect_length = habitat_properties_dataset.shape
transect_count, soil_fields, transect_length = soil_properties_dataset.shape

# Creating the numpy arrays that will store all the information for a single transect
hab_types = numpy.array(transect_length)
hab_properties = numpy.array((habitat_fields, transect_length))
bathymetry = numpy.array(transect_length)
positions = numpy.array((transect_length, 2))
soil_types = numpy.array(transect_length)
soil_properties = numpy.array((soil_fields, transect_length))
indices_limit = numpy.array((transect_length, 2))
coordinates_limits = numpy.array((transect_length, 4))

print('Number of transects', transect_count)
print('Maximum number of habitat fields', habitat_fields)
print('Maximum number of soil fields', soil_fields)
print('Maximum transect length', transect_length)

# Open field indices file, so that we can access a specific field by name
field_indices = json.load(open('field_indices')) # Open field indices info

# Field indices
print('')
print('')

for habitat_type in field_indices:
    habitat = field_indices[habitat_type]
    print('habitat ' + str(habitat_type) + ' has ' + \
          str(len(habitat['fields'])) + ' fields:')

    for field in habitat['fields']:
        print('field ' + str(field) + ' is ' + str(habitat['fields'][field]))
    print('')
#--------------------------------------------------------------
#Run the loop
#--------------------------------------------------------------

nodata = -99999.0

#Initialize matrices
Depth=numpy.zeros((transect_count,transect_length))+nodata
Wave =numpy.zeros((transect_count,transect_length))+nodata
WaterLevel=numpy.zeros((transect_count,transect_length))+nodata
WaterLevel_NoVegetation=numpy.zeros((transect_count,transect_length))+nodata
VeloBottom=numpy.zeros((transect_count,transect_length))+nodata
Undertow=numpy.zeros((transect_count,transect_length))+nodata
SedTrsprt=numpy.zeros((transect_count,transect_length))+nodata

#Read data for each transect, one at a time
for transect in range(1258,1259): #transect_count):
    print('')
    print('transect', transect_count - transect)

    # Extract first and last index of the valid portion of the current transect
    start = indices_limit_dataset[transect,0]   # First index is the most landward point
    end = indices_limit_dataset[transect,1] # Second index is the most seaward point
    # Note: For bad data, the most landard point could be in the ocean or the most
    #   seaward point could be on land!!!
    Length=end-start;Start=start;End=end;
    print('index limits (start, end):', (Start, End))
    
    # Extracting the valid portion (Start:End) of habitat properties
    hab_properties = habitat_properties_dataset[transect,:,Start:End]
    # The resulting matrix is of shape transect_count x 5 x max_transect_length
    # The middle index (1) is the maximum number of habitat fields:
    print('maximum habitat property fields:', hab_properties.shape[1])
    
    unique_types = numpy.unique(hab_types)  # Compute the unique habitats

    #Bathymetry
    bathymetry = bathymetry_dataset[transect,Start:End]
    max_depth = numpy.amax(bathymetry)
    min_depth = numpy.amin(bathymetry)
    Shore=Indexed(bathymetry,0) #locate zero
    MinDepth=Indexed(bathymetry,min_depth)
    print('bathymetry (min, max)', (min_depth, max_depth))
    
    if min_depth>-1 or abs(MinDepth-Shore)<2: #If min depth is too small and there aren't enough points, we don't run
        H=num.zeros(len(bathymetry))
    else: #Run the model
        #------Read habitat
        seagrass = 2
        
        # Load the habitat types along the valid portion (Start:End) of the current transect
        hab_types = habitat_types_dataset[transect,Start:End]
        habitat_types = numpy.unique(hab_types)#Different types of habitats
        habitat_types = habitat_types[habitat_types>=0]#Remove 'nodata' where theres no habitat
        
        positions = positions_dataset[transect,Start:End]
        start = [positions[0][0]]
        start.append(positions[0][1])
        end = [positions[-1][0]]
        end.append(positions[-1][1])
        coordinates_limits = coordinates_limits_dataset[transect,:]
        print('coord limits:', \
              (coordinates_limits[0], coordinates_limits[1]), \
              (coordinates_limits[2], coordinates_limits[3]))
    
        #--Collect vegetation properties 
        #Zero the phys. char
        RootDiam=numpy.zeros(Length);    RootHeight=numpy.zeros(Length);
        RootDens=numpy.zeros(Length);    RootCd=numpy.zeros(Length);
        TrunkDiam=numpy.zeros(Length);    TrunkHeight=numpy.zeros(Length);
        TrunkDens=numpy.zeros(Length);    TrunkCd=numpy.zeros(Length);
        CanopDiam=numpy.zeros(Length);    CanopHeight=numpy.zeros(Length);
        CanopDens=numpy.zeros(Length);    CanopCd=numpy.zeros(Length)
    
        if habitat_types.size: #If there is a habitat in the profile
            seagrass=2
            HabType=[];#Collect the names of the different habitats
            if seagrass in habitat_types:
                HabType.append('Seagrass')
                seagrass_location = numpy.where(hab_types == seagrass)
            
                #Seagrass physical parameters - 'field_indices' dictionary
                if seagrass_location[0].size:
                    Sg_diameter_id = field_indices[str(seagrass)]['fields']['stemdiam']
                    Sg_diameters = hab_properties[Sg_diameter_id][seagrass_location]
                    mean_stem_diameter = numpy.average(Sg_diameters)
                    print('   Seagrass detected. Mean stem diameter: ' + \
                        str(mean_stem_diameter) + ' m')
            
                    Sg_height_id = field_indices[str(seagrass)]['fields']['stemheight']
                    Sg_height = hab_properties[Sg_height_id][seagrass_location]
                    mean_stem_height = numpy.average(Sg_height)
                    print('                                    Mean stem height: ' + \
                        str(mean_stem_height) + ' m')
                    
                    Sg_density_id = field_indices[str(seagrass)]['fields']['stemdensty']
                    Sg_density = hab_properties[Sg_density_id][seagrass_location]
                    mean_stem_density = numpy.average(Sg_density)
                    print('                                    Mean stem density: ' + \
                          str(mean_stem_density) + ' #/m^2')
                    
                    Sg_drag_id = field_indices[str(seagrass)]['fields']['stemdrag']
                    Sg_drag = hab_properties[Sg_drag_id][seagrass_location]
                    mean_stem_drag = numpy.average(Sg_drag)
                    print('                                    Mean stem drag: ' + \
                        str(mean_stem_drag) )
                    
                    RootDiam[seagrass_location]=Sg_diameters
                    RootHeight[seagrass_location]=Sg_height
                    RootDens[seagrass_location]=Sg_density
                    RootCd[seagrass_location]=Sg_drag
                    
            print('unique habitat types:', HabType)
                
        #Collect reef properties
        
        #Collect Oyster Reef properties
        Oyster={}
        
        #Soil types and properties   
        soil_types = soil_types_dataset[transect,Start:End]
        soil_properties = soil_properties_dataset[transect,:,Start:End]
        print('maximum soil property fields:', soil_properties.shape[1])
        print('soil types', numpy.unique(soil_types)) #, soil_types)
        
        #Prepare to run the model
        dx=20;
        import CPf_SignalSmooth as SignalSmooth
        smoothing_pct=10.0
        smoothing_pct=smoothing_pct/100;
        
        #Resample the input data
        Xold=range(0,dx*len(bathymetry),dx)
        Xnew=range(0,Xold[-1]+1)
        length=len(Xnew)
        from scipy import interpolate
        fintp=interpolate.interp1d(Xold,bathymetry, kind='linear')
        bath=fintp(Xnew)
        bath_sm=SignalSmooth.smooth(bath,len(bath)*smoothing_pct,'hanning') 
        shore=Indexed(bath_sm,0) #Locate zero in the new vector
        
        fintp=interpolate.interp1d(Xold,RootDiam, kind='nearest')
        RtDiam=fintp(Xnew)
        fintp=interpolate.interp1d(Xold,RootHeight, kind='nearest')
        RtHeight=fintp(Xnew)
        fintp=interpolate.interp1d(Xold,RootDens, kind='nearest')
        RtDens=fintp(Xnew)
        fintp=interpolate.interp1d(Xold,RootCd, kind='nearest')
        RtCd=fintp(Xnew)
        
        fintp=interpolate.interp1d(Xold,TrunkDiam, kind='nearest')
        TkDiam=fintp(Xnew)
        fintp=interpolate.interp1d(Xold,TrunkHeight, kind='nearest')
        TkHeight=fintp(Xnew)
        fintp=interpolate.interp1d(Xold,TrunkDens, kind='nearest')
        TkDens=fintp(Xnew)
        fintp=interpolate.interp1d(Xold,TrunkCd, kind='nearest')
        TkCd=fintp(Xnew)
    
        fintp=interpolate.interp1d(Xold,CanopDiam, kind='nearest')
        CpDiam=fintp(Xnew)
        fintp=interpolate.interp1d(Xold,CanopHeight, kind='nearest')
        CpHeight=fintp(Xnew)
        fintp=interpolate.interp1d(Xold,CanopDens, kind='nearest')
        CpDens=fintp(Xnew)
        fintp=interpolate.interp1d(Xold,CanopCd, kind='nearest')
        CpCd=fintp(Xnew)
    
        hab_types[hab_types==nodata]=-1
        fintp=interpolate.interp1d(Xold,hab_types, kind='nearest')
        Sr=fintp(Xnew)
        
        #Check to see if we need to flip the data
        flip=0
        if bath_sm[0]>bath_sm[-1]:
            bath_sm=bath_sm[::-1];        flip=1
            RtDiam=RtDiam[::-1];        RtHeight=RtHeight[::-1]
            RtDens=RtDens[::-1];        RtCd=RtCd[::-1]
            TkDiam=TkDiam[::-1];        TkHeight=TkHeight[::-1]
            TkDens=TkDens[::-1];        TkCd=TkCd[::-1]
            CpDiam=CpDiam[::-1];        CpHeight=CpHeight[::-1]
            CpDens=CpDens[::-1];        CpCd=CpCd[::-1]
            Sr=Sr[::-1]
            
        #Store hab char. into dic
        PlantsPhysChar={};Roots={};Trunks={};Canops={}
        Roots["RootDiam"]=RtDiam;    Roots["RootHeight"]=RtHeight
        Roots["RootDens"]=RtDens;    Roots["RootCd"]=RtCd
        Trunks["TrunkDiam"]=TkDiam;    Trunks["TrunkHeight"]=TkHeight
        Trunks["TrunkDens"]=TkDens;    Trunks["TrunkCd"]=TkCd
        Canops["CanopDiam"]=CpDiam;    Canops["CanopHeight"]=CpHeight
        Canops["CanopDens"]=CpDens;    Canops["CanopCd"]=CpCd
        #Final dictionary
        PlantsPhysChar['Roots']=Roots.copy()
        PlantsPhysChar['Trunks']=Trunks.copy()
        PlantsPhysChar['Canops']=Canops.copy()
        PlantsPhysChar['Oyster']=Oyster.copy()
        
        #Define friction coeff
        #   0 = kelp #   1 = eelgrass #   2 = underwater structure/oyster reef
        #   3 = coral reef #   4 = levee #   5 = beach #   6 = seawall
        #   7 = marsh #   8 = mangrove #   9 = terrestrial structure
        Cf=numpy.zeros(length)+.01
        if flip==1:
            Cf=Cf[::-1]
            
        #Compute Wave Height        
        Xnew=num.array(Xnew)
        H,Eta,Etanv,Ubot,Ur,Kt,Ic,Hm,other=WaveRegenWindCD(Xnew,bath_sm,Surge,Ho,To,Uo,Cf,Sr,PlantsPhysChar)
    
        #Compute maximum wave height
        k,C,Cg=Fast_k(To,-bath_sm)
        Hmx=0.1*(2.0*pi/k)*tanh((-bath_sm+Surge)*k);#Max wave height - Miche criterion
    
        #Wave Breaking information
        temp,temp,xb,hb,Hb,temp,temp=Runup_ErCoral(Xnew,-bath_sm,H,Ho,H*0,H*0,To,.2,1.0/10,Sr)
        loc,Hb=FindBreaker(Xnew,-bath_sm,H,To,Sr)
        Transport=nanmean(Ic[loc:-1])
        
        #Flip the vectors back
        if flip==1:
            H=H[::-1];Eta=Eta[::-1];Etanv=Etanv[::-1];
            Ubot=Ubot[::-1];Ur=Ur[::-1];Ic=Ic[::-1]
            h=bath_sm[::-1];X=Xnew[::-1]
            
        #Interpolate back to dx and save in matrix
        lx=len(Xold)
        fintp=interpolate.interp1d(X,h, kind='linear')
        h_save=fintp(Xold)
        Depth[transect,0:lx]=h_save
        
        fintp=interpolate.interp1d(X,H, kind='linear')
        H_save=fintp(Xold)
        Wave[transect,0:lx]=H_save

        fintp=interpolate.interp1d(X,Eta, kind='linear')
        Eta_save=fintp(Xold)
        WaterLevel[transect,0:lx]=Eta_save

        fintp=interpolate.interp1d(X,Etanv, kind='linear')
        Etanv_save=fintp(Xold)
        WaterLevel_NoVegetation[transect,0:lx]=Etanv_save

        fintp=interpolate.interp1d(X,Ubot, kind='linear')
        Ubot_save=fintp(Xold)
        VeloBottom[transect,0:lx]=Ubot_save

        fintp=interpolate.interp1d(X,Ur, kind='linear')
        Ur_save=fintp(Xold)
        Undertow[transect,0:lx]=Ur_save

        fintp=interpolate.interp1d(X,Ic, kind='linear')
        Ic_save=fintp(Xold)
        SedTrsprt[transect,0:lx]=Ic_save
        
        #Compute beach erosion
        Beach=-1;Struct=-1
        if Beach==1:
            g=9.81;rho=1024.0;Gam=0.78;
            TD=Dur;Lo=g*To**2.0/(2.0*pi);
            Co=g*To/(2.0*pi);#deep water phase speed with period
            
            Rs0,Rs1,xb,hb,Hb,Etapr,Hpr=Runup_ErCoral(Xnew,-bath_sm,H,Ho,Eta,Eta,To,A,m,Sr)
            #Quick Estimate
            TS=(320.0*(Hb**(3.0/2)/(g**.5*A**3.0))/(1.0+hb/(BermH_P+DuneH_P)+(m*xb)/hb))/3600.0;#erosion response time scale ).
            BetaKD=2.0*pi*(TS/TD)
            expr="num.exp(-2.0*x/BetaKD)-num.cos(2.0*x)+(1.0/BetaKD)*num.sin(2.0*x)" # solve this numerically
            fn=eval("lambda x: "+expr)
            z=FindRootKD(fn,pi,pi/2,BetaKD) # find zero in function,initial guess from K&D
            Ro,Rinfo,m0=Erosion_Quick(Ho,To,Surge[-1]+Rs0,BermH_P,DuneH_P,BermW_P,Dur,m,A,z) #Quick estimate
            
            #Erosion using waves
            TS=(320.0*(Hb**(3.0/2)/(g**.5*A**3.0))/(1.0+hb/(BermH_P+DuneH_P)+(m*xb)/hb))/3600.0;#erosion response time scale ).
            BetaKD=2.0*pi*(TS/TD)
            z=FindRootKD(fn,pi,pi/2,BetaKD) # find zero in function,initial guess from K&D
            R,Rinf,mo=ErosionFunction(A,m,xb,hb,Hb,Surge1[-1]+Rs0,BermH_P,DuneH_P,BermW_P,Dur,z)
            
            #Present
            Rsp0,Rsp1,xb1,hb1,Hb1,Etapr1,Hpr1=Runup_ErCoral(X1,-Z1,Hp,Ho,Eta0,Etap,To,A,m,Sr1)
            TS=(320.0*(Hb1**(3.0/2)/(g**.5*A**3.0))/(1.0+hb1/(BermH_P+DuneH_P)+(m*xb1)/hb1))/3600.0;#erosion response time scale ).
            BetaKD=2.0*pi*(TS/TD)
            z=FindRootKD(fn,pi,pi/2,BetaKD) # find zero in function,initial guess from K&D
            R1,Rinf1,m1=ErosionFunction(A,m,xb1,hb1,Hb1,Surge1[-1]+Rsp1,BermH_P,DuneH_P,BermW_P,Dur,z)
    
            Ro=round(Ro,2);R=round(R,2);R1=round(R1,2);
            
        #Compute mud scour
        if Beach==0:
            if Mgloc1.any(): #If there is any mangroves present at the site
                temp1=Mgloc1[0];
            else:
                temp1=-1
            
            Mudy1=[];#Location of the muddy bed
            if (temp1)>=0 or (temp2)>=0:
                MudBeg=min(temp1,temp2)
                Mudy1=arange(MudBeg,Xend1)
                
            MErodeVol1=-1; #No mud erosion        
            if len(Mudy1)>0:#Calculate mud erosion if there's a muddy bed
                Ubp=array(Ubp);
                Retreat1,Trms1,Tc1,Tw1,Te=MudErosion(Ubp[Mudy1]*0,Ubp[Mudy1],-Z1[Mudy1],To,me,Cm)
                ErodeLoc=find(Trms1>Te[0]); # indices where erosion rate greater than Threshold
                MErodeLen1=len(ErodeLoc) # erosion rate greater than threshold at each location shoreward of the shoreline (pre-management)
                if any(ErodeLoc)>0:
                    MErodeVol1=trapz(Retreat1[ErodeLoc]/100.0,Mudy1[ErodeLoc],1.) # volume of mud eroded shoreward of the shoreline (m^3/m)
                else:
                    MErodeVol1=0
                MErodeVol1=round(MErodeVol1,2)
                gp.addmessage('MudScour_Present='+str(MErodeVol1)+' m^3/m')
    
        if Struct==1:
            #Current conditions
            Qp=150;htoe=round(-Z1[-1]+Etap[-1],1);Hwp=htoe
            while Qp>10:
                Hwp=Hwp+.1;
                temp,Qp,msg=SeawallOvertop(Hp[-1],To,htoe,Hwp)
            Fp=ForceSeawall(Hp[-1],To,-Z1[-1])
            Fp=round(Fp,2);Hwp=round(Hwp,2)
            gp.addmessage('Wall_Present='+str(Hwp)+' m; Force on wall='+str(Fp)+' kN/m')
    

# Saving data in HDF5
f = h5py.File('outputs.h5', 'w') # Create new HDF5 file

# Creating the placeholders that will hold the matrices
Depth_dataset = \
    f.create_dataset("Depth", Depth.shape, \
        compression = 'gzip', fillvalue = nodata)
Wave_dataset = \
    f.create_dataset("Wave", Wave.shape, \
        compression = 'gzip', fillvalue = nodata)
WaterLevel_dataset = \
    f.create_dataset("WaterLevel", WaterLevel.shape, \
        compression = 'gzip', fillvalue = nodata)
WaterLevel_NoVegetation_dataset = \
    f.create_dataset("WaterLevel_NoVegetation", WaterLevel_NoVegetation.shape, \
        compression = 'gzip', fillvalue = nodata)
VeloBottom_dataset = \
    f.create_dataset("VeloBottom", VeloBottom.shape, \
        compression = 'gzip', fillvalue = nodata)
Undertow_dataset = \
    f.create_dataset("Undertow", Undertow.shape, \
        compression = 'gzip', fillvalue = nodata)
SedTrsprt_dataset = \
    f.create_dataset("SedTrsprt", SedTrsprt.shape, \
        compression = 'gzip', fillvalue = nodata)

# Saving the matrices on file
Depth_dataset[...] = Depth[...]
Wave_dataset[...] = Wave[...]
WaterLevel_dataset[...] = WaterLevel[...]
WaterLevel_NoVegetation_dataset[...] = WaterLevel_NoVegetation[...]
VeloBottom_dataset[...] = VeloBottom[...]
Undertow_dataset[...] = Undertow[...]
SedTrsprt_dataset[...] = SedTrsprt[...]

# Close your file at the end, or it could be invalid.
f.close()
