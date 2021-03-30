
import numpy as np
import scipy as sp
import matplotlib  
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle

from matplotlib.patches import Wedge

import matplotlib.transforms as transforms


def rectagular_obstacle(left_corner_x,left_corner_y,width,height,ax,angle=0.0,facecolor='none',**kwargs):
    addrectangle = Rectangle((left_corner_x,left_corner_y),width=width,height=height,angle=angle,facecolor=facecolor, **kwargs) 
    return ax.add_patch(addrectangle)

def circular_obstacle(x, y, radius, ax, facecolor='none', **kwargs):
    
    """
    Create a plot of the obstacle.

    Parameters
    ----------
    x, y : is the center of the obstacle
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    Returns
    -------
    matplotlib.patches.circle

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    center_x = x
    center_y = y 
    add_circle = Circle((center_x,center_y),radius,facecolor=facecolor, **kwargs) 
    return ax.add_patch(add_circle)

def spacesim_2d(x,y,theta,ax,facecolor='none',**kwargs):
    transf = transforms.Affine2D() \
    .rotate_deg(np.rad2deg(theta))

    addrectangle = Rectangle((x-0.3,y-0.3),width=0.6,height=0.6,facecolor=facecolor, **kwargs)      

    addrectangle.set_transform(transf + ax.transData)
    return ax.add_patch(addrectangle)

def add_thrusters(x,y,theta,ax,facecolor='none',**kwargs):
    theta_deg = np.rad2deg(theta)
    
    # tranfs = transforms.Affine2D().rotate_deg(theta_deg)
    # print(tranfs)

    R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

    t1 = R@np.array([x+0.3,y-0.2])
    t2 = R@np.array([x+0.3,y+0.2])
    t3 = R@np.array([x+0.2,y +0.3])
    t4 = R@np.array([x-0.2,y+0.3])
    t5 = R@np.array([x-0.3,y-0.2])
    t6 = R@np.array([x-0.3,y+0.2])
    t7 = R@np.array([x+0.2,y-0.3])
    t8 = R@np.array([x-0.2,y-0.3])

    wedge1 = Wedge((t1[0],t1[1]),0.03, theta_deg-20,theta_deg+20,facecolor=facecolor)
    wedge2 = Wedge((t2[0],t2[1]),0.03, theta_deg-20,theta_deg+20,facecolor=facecolor)

    wedge3 = Wedge((t3[0],t3[1]),0.03, theta_deg+90-20,theta_deg+90+20,facecolor=facecolor)
    wedge4 = Wedge((t4[0],t4[1]),0.03, theta_deg+90-20,theta_deg+90+20,facecolor=facecolor)
    
    wedge5 = Wedge((t5[0],t5[1]),0.03, theta_deg+180-20,theta_deg+180+20,facecolor=facecolor)
    wedge6 = Wedge((t6[0],t6[1]),0.03, theta_deg+180-20,theta_deg+180+20,facecolor=facecolor)
    
    wedge7 = Wedge((t7[0],t7[1]),0.03, theta_deg+270-20,theta_deg+270+20,facecolor=facecolor)
    wedge8 = Wedge((t8[0],t8[1]),0.03, theta_deg+270-20,theta_deg+270+20,facecolor=facecolor)
    
    ax.add_patch(wedge1)
    ax.add_patch(wedge2)

    ax.add_patch(wedge3)
    ax.add_patch(wedge4)
    
    ax.add_patch(wedge5)
    ax.add_patch(wedge6)

    ax.add_patch(wedge7)
    ax.add_patch(wedge8)
    
    return 1


# Confidence ellipse using pearson coefficients 

def confidence_ellipse(m_x, m_y, cov, ax, n_std=2.0, facecolor='none', **kwargs):
    
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    mean_x = m_x
    mean_y = m_y

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std


    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x,mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)




if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    
    _, ax1 = plt.subplots(1, 1, sharey=True, figsize = (6,4))
    x = 0 
    y = 0 
    theta = 3.14/4

    spacesim_2d(x,y,theta,ax1,facecolor='blue',alpha=0.3)
    add_thrusters(x,y,theta,ax1,facecolor='red')

    plt.show()
    
    