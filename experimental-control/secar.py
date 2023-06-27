import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_drift(l): 
    # l: effective length
    # g: gamma
    m = np.eye(5)
    m[0,1] = l  
    m[2,3] = l  
    return( m )

def create_quadH(l, a, b, brho): 
    # l: effective length
    # a: radius of aperture
    # b: field at radius a
    # brho: magnetic rigidity of central trajectory
    k = np.sqrt( b/a*1/brho )
    m = np.eye(5)
    m[0,0] = np.cos(k*l)      
    m[0,1] = 1/k*np.sin(k*l)  
    m[1,0] = -k*np.sin(k*l) 
    m[1,1] = np.cos(k*l)   
    m[2,2] = np.cosh(k*l)      
    m[2,3] = 1/k*np.sinh(k*l)  
    m[3,2] = k*np.sinh(k*l) 
    m[3,3] = np.cosh(k*l)
    return( m )

def create_quadV(l, a, b, brho): 
    # l: effective length
    # a: radius of aperture
    # b: field at radius a
    # brho: magnetic rigidity of central trajectory
    k = np.sqrt( b/a*1/brho )
    m = np.eye(5)
    m[0,0] = np.cosh(k*l)      
    m[0,1] = 1/k*np.sinh(k*l)  
    m[1,0] = k*np.sinh(k*l) 
    m[1,1] = np.cosh(k*l)   
    m[2,2] = np.cos(k*l)      
    m[2,3] = 1/k*np.sin(k*l)  
    m[3,2] = -k*np.sin(k*l) 
    m[3,3] = np.cos(k*l)
    return( m )

def create_dipole(rho, theta):
    # rho: dipole radius
    # theta: dipole angle
    l = theta*rho # length of the arc
    m = np.eye(5)
    m[0,0] = np.cos(theta)      
    m[0,1] = rho*np.sin(theta) 
    m[0,4] = rho*(1-np.cos(theta))
    m[1,0] = -1/rho*np.sin(theta) 
    m[1,1] = np.cos(theta)
    m[1,4] = np.sin(theta)
    m[2,3] = l
    return (m)
    
def create_WF(gamma, l, hB, hE = 1/7):
    # l: WF length
    # gamma: E/mc^2
    h = 5000*(hE + hB)
    xi = np.sqrt((hE/gamma)**2 + h**2)
    zi = hE/gamma**2 + h
    m = np.eye(5)
    m[0,0] = np.cos(l*xi)
    m[0,1] = 1/xi*np.sin(l*xi)
    m[0,4] = (zi/xi**2)*(1-np.cos(l*xi))
    m[1,0] = -xi*np.sin(l*xi)
    m[1,1] = np.cos(l*xi)
    m[1,4] = (zi/xi)*np.sin(l*xi)
    m[2,3] = l
    return (m)

def secar(brho, gamma, NumberParticles = 100, rhoE = 7, Q1s = 1, Q2s = 1, Q3s = 1, Q4s = 1, Q5s = 1, Q6s = 1, \
          Q7s = 1, Q8s = 1,\
          xmax = 0.00075, axmax = 0.025, ymax = 0.00075, aymax = 0.025, dE = 0.031, dQ = 0, dM = 0): 

    rays = []
    
    # "Nominal optics" used with nomBrho 
    nomBrho = 0.800081
    Q1 = 0.39773*Q1s    # Vertical focus
    Q2 = 0.219352*Q2s   # Horizontal focus
    H1 = 0.0103065
    Q3 = 0.242872*Q3s   # Horizontal focus
    Q4 = 0.24756*Q4s    # Vertical focus
    Q5 = 0.112391*Q5s   # Horizontal focus
    H2 = 0.010507   
    Q6 = 0.181632*Q6s   # Horizontal focus
    Q7 = 0.030022*Q7s   # Vertical focus
    H3 = -0.0083745 
    # O1 = 0.031283
    Q8 = 0.20000*Q8s   # Horizontal focus
    scaleBrho = brho/nomBrho

    x = np.random.uniform( low = -xmax, high = xmax, size=NumberParticles)
    ax = np.random.uniform( low = -axmax, high = axmax, size=NumberParticles)
    y = np.random.uniform( low = -ymax, high = ymax, size=NumberParticles)
    ay = np.random.uniform( low = -aymax, high = aymax, size=NumberParticles)
    
    if NumberParticles != 1:
        dE = np.random.uniform( low = -dE, high = dE, size=NumberParticles)
    else:
        dE = [dE]

    particle = np.asmatrix( [ (x[i], ax[i], y[i], ay[i], dE[i]) for i in range(len(x)) ] ).T  
    rays.append(particle)

    dl1 = create_drift(l = 0.800527)   #1
    rays.append(dl1 @ rays[-1])
    q1 =  create_quadV(l = 0.2498, a = 0.055, b = Q1*scaleBrho, brho = brho)  # Vertical focus  #2
    rays.append(q1 @ rays[-1])
    dl2 = create_drift(l = 0.190490)   #3
    rays.append(dl2 @ rays[-1])
    q2 =  create_quadH(l = 0.2979, a = 0.068, b = Q2*scaleBrho, brho = brho)  # Horizontal focus #4
    rays.append(q2 @ rays[-1])
    dl3 = create_drift(l = 0.581038)   #5
    rays.append(dl3 @ rays[-1])
    b1 =  create_dipole(rho = 1.25, theta = np.radians(22.51145))  #6
    rays.append(b1 @ rays[-1])
    dl4 = create_drift(l = 0.999778)   #7
    rays.append(dl4 @ rays[-1])
    b2 =  create_dipole(rho = 1.25, theta = np.radians(22.5121))   #8
    rays.append(b2 @ rays[-1])
    dl5 = create_drift(l = 0.769867)   #9
    rays.append(dl5 @ rays[-1])
    dl6 = create_drift(l = 0.398384 + 0.263)   #10 + Hex1
    rays.append(dl6 @ rays[-1])
    dl7 = create_drift(l = 0.268763)   #11
    rays.append(dl7 @ rays[-1])
    q3 =  create_quadH(l = 0.3499, a = 0.11, b = Q3*scaleBrho, brho = brho)   # Horizontal focus  #12
    rays.append(q3 @ rays[-1])
    dl8 = create_drift(l = 0.351390)   #13
    rays.append(dl8 @ rays[-1])
    q4 =  create_quadV(l = 0.3467, a = 0.08, b = Q4*scaleBrho, brho = brho)   # Vertical focus    #14
    rays.append(q4 @ rays[-1])
    dl9 = create_drift(l = 0.213664)   #15
    rays.append(dl9 @ rays[-1])
    q5 =  create_quadH(l = 0.3466, a = 0.06, b = Q5*scaleBrho, brho = brho)   # Horizontal focus  #16
    rays.append(q5 @ rays[-1])
    dl10 = create_drift(l = 0.146731)  #17
    rays.append(dl10 @ rays[-1])
    # FP1 viewer -------------------------------------
    dl11 = create_drift(l = 0.185)     #18
    rays.append(dl11 @ rays[-1])
    dl12 = create_drift(l = 0.169301)  #19
    rays.append(dl12 @ rays[-1])
    b3 =  create_dipole(rho = 1.25, theta = np.radians(22.5321))   #20
    rays.append(b3 @ rays[-1])
    dl13 = create_drift(l = 0.509073)  #21
    rays.append(dl13 @ rays[-1])
    b4 =  create_dipole(rho = 1.25, theta = np.radians(22.5807))   #22
    rays.append(b4 @ rays[-1])
    dl14 = create_drift(l = 0.297393 + 0.264)  #23 + Hex2
    rays.append(dl14 @ rays[-1])
    # Hex2
    dl15 = create_drift(l = 0.270097)  #24
    rays.append(dl15 @ rays[-1])
    dl16 = create_drift(l = 0.268113)  #25
    rays.append(dl16 @ rays[-1])
    q6 =  create_quadH(l = 0.3398, a = 0.14, b = Q6*scaleBrho, brho = brho)   # Horizontal focus   #26
    rays.append(q6 @ rays[-1])
    dl17 = create_drift(l = 0.199837)  #27
    rays.append(dl17 @ rays[-1])
    q7 =  create_quadV(l = 0.3401, a = 0.13, b = Q6*scaleBrho, brho = brho)   # Vertical focus     #28
    rays.append(q7 @ rays[-1])
    dl18 = create_drift(l = 0.499738)  #29
    rays.append(dl18 @ rays[-1])
    hE = 1/rhoE
    wf1 = create_WF(gamma = gamma, l = 2.365, hB = -hE*(1+dQ)/(1+dM), hE = hE)  #30
    rays.append(wf1 @ rays[-1])
    dl19 = create_drift(l = 0.498516)  #31
    rays.append(dl19 @ rays[-1])
    # Hex3
    q8 =  create_quadH(l = 0.263, a = 0.14, b = Q8*scaleBrho, brho = brho)   # Horizontal focus   #32
    rays.append(q8 @ rays[-1])
    dl20 = create_drift(l = 0.277156 + 0.262)  #33
    rays.append(dl20 @ rays[-1])
    # Oct1
    dl21 = create_drift(l = 0.5)  #34
    rays.append(dl21 @ rays[-1])
    # FP2 viewer -------------------------------------
        
    return(rays)

class viewer:
    def __init__(self, bandwidth, locations=None):
        super(viewer, self).__init__()
        self.bandwidth = bandwidth
        self.locations = locations

    def forward(self, samples, locations=None):
        if locations is None:
            locations = self.locations
        assert samples.shape[-1] == locations.shape[-1]

        # Make copies of all samples for each location
        all_samples = np.reshape(samples, samples.shape + (1,) * len(locations.shape[:-1]))
        diff = np.linalg.norm(
            all_samples - np.moveaxis(locations, -1, 0),
            axis=-len(locations.shape[:-1]) - 1
        )
        out = np.sum(np.exp(-diff ** 2 / self.bandwidth ** 2),axis=len(samples.shape)-2)
        norm = np.sum(np.reshape(out, (-1, out.shape[-1])), axis=-1)
        return out / norm.reshape(-1, *(1,)*(len(locations.shape)-1))
    
def plotOptics(recoils, unreacted = None):
    lengths = [0.800527, 0.2498, 0.190490, 0.2979, 0.581038, 1.25*np.radians(22.51145), 0.999778, \
                1.25*np.radians(22.5121), 0.769867, 0.398384 + 0.263, 0.268763, 0.3499, 0.351390, 0.3467, 0.213664, \
                0.3466, 0.146731, 0.185, 0.169301, 1.25*np.radians(22.5321), 0.509073, \
                1.25*np.radians(22.5807), 0.297393 + 0.264, 0.270097, 0.268113, 0.3398, 0.199837, \
                0.3401, 0.499738, 2.365, 0.498516, 0.263, 0.277156 + 0.262, 0.5]  # Optical element sizes
    distanceFromTarget = [0]
    for i in range(len(lengths)):
        distanceFromTarget.append(distanceFromTarget[-1] + lengths[i])
    
    plt.rc('font', size=16)
    fig, ax = plt.subplots(2, 1, figsize=(16, 8))
    for k in range( len(np.asarray(recoils[0][0])[0]) ):
        X = [recoils[j][0,k] for j in range(len(distanceFromTarget))] 
        Y = [recoils[j][2,k] for j in range(len(distanceFromTarget))] 
        ax[0].plot(distanceFromTarget,X, c='gray', alpha = 0.5)
        ax[1].plot(distanceFromTarget,Y, c='gray', alpha = 0.5)
    if unreacted is not None:
        X_unreacted = [unreacted[j][0,0] for j in range(len(distanceFromTarget))] 
        ax[0].plot(distanceFromTarget, X_unreacted, c='black', linewidth=5.0)
        
    for i in range(2):
        # dipoles
        ax[i].axvline(x=distanceFromTarget[6], color='blue', linestyle='--')
        ax[i].axvline(x=distanceFromTarget[8], color='blue', linestyle='--')
        ax[i].axvline(x=distanceFromTarget[20], color='blue', linestyle='--')
        ax[i].axvline(x=distanceFromTarget[22], color='blue', linestyle='--')
        # quads
        ax[i].axvline(x=distanceFromTarget[2], color='red', linestyle='--')
        ax[i].axvline(x=distanceFromTarget[4], color='red', linestyle='--')
        ax[i].axvline(x=distanceFromTarget[12], color='red', linestyle='--')
        ax[i].axvline(x=distanceFromTarget[14], color='red', linestyle='--')
        ax[i].axvline(x=distanceFromTarget[16], color='red', linestyle='--')
        ax[i].axvline(x=distanceFromTarget[26], color='red', linestyle='--')
        ax[i].axvline(x=distanceFromTarget[28], color='red', linestyle='--')
        ax[i].axvline(x=distanceFromTarget[32], color='red', linestyle='--')
        # WF1
        ax[i].axvline(x=distanceFromTarget[30], color='yellow', linestyle='--')
        ax[i].axvline(x=distanceFromTarget[17], color='black')   # FP1
        ax[i].axvline(x=distanceFromTarget[34], color='black')   # FP2

    ax[0].text(1, 0.2, "Dipole", ha='center', va='bottom', color = 'blue')
    ax[0].text(1, 0.12, "Quad", ha='center', va='bottom', color = 'red')
    ax[0].text(1, -0.12, "Wien Filter", ha='center', va='bottom', color = 'yellow')
    ax[0].text(1, -0.2, "Focal plane", ha='center', va='bottom', color = 'black')
    ax[0].set_ylim(-0.3, 0.3)
        
    #ax[0].set_xlabel('Distance from target [m]')
    ax[0].set_ylabel('Horizontal beam size [m]')
    ax[1].set_xlabel('Distance from target [m]')
    ax[1].set_ylabel('Vertical beam size [m]')
    ax[1].set_ylim(-0.2, 0.2)

    plt.show()
    
def plotViewers(rays):
    fp1 = rays[17]
    fp2 = rays[34]

    # Generate particle images
    fig, ax = plt.subplots(2, 2, figsize=(16, 8))
    a = 0.5

    bin_x = np.linspace(-a,a,60)
    bin_y = np.linspace(-a,a,60)
    ax[0,0].hist2d(np.asarray(fp1[0])[0], np.asarray(fp1[2])[0], bins = [bin_x, bin_y])
    ax[0,0].set_xlim(-a, a)
    ax[0,0].set_ylim(-a, a)
    ax[0,0].set_title('FP1 Raw Viewer')
    ax[0,1].hist2d(np.asarray(fp2[0])[0], np.asarray(fp2[2])[0], bins = [bin_x, bin_y])
    ax[0,1].set_xlim(-a, a)
    ax[0,1].set_ylim(-a, a)
    ax[0,1].set_title('FP2 Raw Viewer')

    # Generate viewer images
    X_, Y_ = np.mgrid[-a:a:100j, -a:a:100j]
    locations = np.vstack([X_.ravel(), Y_.ravel()]).T

    bandwidth = 0.05
    viewerClass = viewer(bandwidth = bandwidth, locations = locations)
    points = np.array( [ [fp1[0,i], fp1[2,i]] for i in range(len(np.asarray(fp1[0])[0]))] )
    viewerFP1 = viewerClass.forward(points)

    viewerClass = viewer(bandwidth = bandwidth, locations = locations)
    points = np.array( [ [fp2[0,i], fp2[2,i]] for i in range(len(np.asarray(fp1[0])[0]))] )
    viewerFP2 = viewerClass.forward(points)

    sns.heatmap(np.reshape(viewerFP1, X_.shape).T, cmap='coolwarm', ax=ax[1,0])
    ax[1,0].set_title('FP1 HeatMap')
    sns.heatmap(np.reshape(viewerFP2, X_.shape).T, cmap='coolwarm', ax=ax[1,1])
    ax[1,1].set_title('FP2 HeatMap')
    plt.show()

def get_width(recoils):
    w1 = max(np.asarray(recoils[17][0])[0]) - min(np.asarray(recoils[17][0])[0])
    w2 = max(np.asarray(recoils[34][0])[0]) - min(np.asarray(recoils[34][0])[0])
    return w1, w2

def get_resolving_power(recoils, unreacted):
    w1, w2 = get_width(recoils)
    R1 = np.abs(np.asarray(unreacted[17][0])[0,0])/w1
    R2 = np.abs(np.asarray(unreacted[34][0])[0,0])/w2
    return R1, R2