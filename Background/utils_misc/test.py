
# coding: utf-8

# In[12]:


#>>>python test.py testDict.testDimQuant+=2 testDict.testNonDimQuant*=2 testDict.testBoolQuant = True


# In[1]:


#this does't actually need to be protected. More a reminder it's an interim measure
import sys
try:
    sys.path.append('../../')
except:
    pass


# In[ ]:


import numpy as np
from easydict import EasyDict as edict
import pint

#import UWsubduction as usub

import UWsubduction.params as params 
import UWsubduction.utils as utils

ur = params.u


# In[ ]:


testDict = edict({})


# In[ ]:


testDict.testDimQuant = 1.1*ur.kilometer
testDict.testNonDimQuant = 2.2
testDict.testBoolQuant = False


# In[ ]:


print('default values:')
for k in testDict.keys():
    print(testDict[k])
#print(testDict.testDimQuant,testDict.testNonDimQuant, testDict.testBoolQuant)


# In[26]:


sysArgs = sys.argv
utils.easy_args(sysArgs, testDict)


# In[27]:


print('updatedvalues:')
for k in testDict.keys():
    print(testDict[k])

if testDict.testBoolQuant == True:
    print('and True in the Python sense')
elif testDict.testBoolQuant == False:
     print('False in the Python sense')
else:
    print 'not resolving this', type(testDict.testBoolQuant)
    

