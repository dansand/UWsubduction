
# coding: utf-8

# In[ ]:

###Example....
#>>>python test.py dp.arg3*=4 dp.arg2='terry'
#


# In[9]:

from unsupported_dan.easymodels import easy_args
import sys
from easydict import EasyDict as edict


# In[10]:

easy_args


# In[8]:

sysArgs = sys.argv
dp = edict({})

dp.arg2='gary'
dp.arg3=2.0
dp.arg4=False

print('default values:')
print(dp.arg2, dp.arg3, dp.arg4)
easy_args(sysArgs, dp)
print('updated values:')
print(dp.arg2, dp.arg3, dp.arg4)

if dp.arg4 == True:
    print('and True in the Python sense')


# In[ ]:
