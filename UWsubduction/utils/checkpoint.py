import os
import sys
import pickle
import underworld as uw



class checkpoint:

    """
    Trying to be a bit more clever about checkpointing

    ...

    notes: the whole uwContext thing. Is is a good idea?
    It just allows us to call through to mpi4py,
    so we can check for rank, set up barriers, etc.

    """

    def __init__(self, savepath, loadpath='default', uwContext = uw):

        assert uwContext, "no underworld"
        self.objDict = {}
        self.dictDict = {}
        self.measuresDict = {}

        state = {'time':0., 'step':0}
        self.addDict(state, 'state')

        self.restart = False
        self.loadpath = '' #is this wise?
        self.uwContext = uwContext
        self.savepath = savepath


        # make directories if they don't exist
        if self.uwContext.rank()==0:
            if not os.path.isdir(self.savepath):
                os.makedirs(self.savepath)
        uwContext.barrier()

        #check the savepath for any subdirs representing checkpoints - default behav.
        if loadpath == 'default':
            d=self.savepath
            subdirs = filter(lambda x: os.path.isdir(os.path.join(d, x)), os.listdir(d))
            subdirs.sort(key=float) #no need for the natsort module, can phase it out elsewhere
            if subdirs:
                lpstring = self.savepath + subdirs[-1]
                self.loadpath = lpstring
                self.restart = True
                #load state dictionary - all procs can do this for now,
                #Though it may be better to restrict to rank 0
                try:
                    with open(os.path.join(self.loadpath, 'state.pkl'), 'rb') as fp:
                        self.dictDict['state'] = pickle.load(fp)
                except:
                    print("could no load saved state info.")


        #check the provided load path, will raise and error if directory doesn't exist / empty
        #does not check that a non-empty directory contains valid objects
        elif isinstance(loadpath , basestring):
            assert os.listdir(loadpath), "directory provided is empty"
            self.loadpath = loadpath
            self.restart = True
            #load state dictionary - all procs can do this for now,
            #Though it may be better to restrict to rank 0
            try:
                with open(os.path.join(self.loadpath, 'state.pkl'), 'rb') as fp:
                        self.dictDict['state'] = pickle.load(fp)
            except:
                print("Could no load saved state info. Missing or corrupt 'state.pkl' file. ")

    def updateState(self, step, time):

        self.dictDict['state']['step'] = step
        self.dictDict['state']['time'] = time


    def time(self):
        return self.dictDict['state']['time']

    def step(self):
        return self.dictDict['state']['step']

    def addDict(self, _dict, name):
        #check if obj has a save method
        #assert hasattr(obj, 'save'), "object provided has no save method"


        self.dictDict[name] = _dict


    def addObject(self, obj, name):
        #check if obj has a save method
        assert hasattr(obj, 'save'), "object provided has no save method"


        self.objDict[name] = obj


    def saveDicts(self, step,  time, savepath = 'default', suffix = '.pkl'):
        #update state
        self.updateState(step, time)



        #will usually save in the self.savepath + 'step'
        #but can overide and put the whole thing anywhere
        if savepath == 'default':
            actualpath = os.path.join(self.savepath,str(self.step()))
        elif isinstance(savepath , basestring):
            actualpath = savepath

        # make directories if they don't exist
        if self.uwContext.rank()==0:
            if not os.path.isdir(actualpath):
                os.makedirs(actualpath)

            #Save Dicts
            for key, val in self.dictDict.items():
                #val.save(os.path.join(actualpath,key + suffix))
                with open(os.path.join(actualpath, key + suffix), 'wb') as fp:
                    pickle.dump( val , fp)

    def saveObjs(self, step,  time, savepath = 'default', suffix = '.h5'):

        #update state
        self.updateState(step, time)

        #will usually save in the self.savepath + 'step'
        #but can overide and put the whole thing anywhere
        if savepath == 'default':
            actualpath = os.path.join(self.savepath,str(self.step()))
        elif isinstance(savepath , basestring):
            actualpath = savepath

        # make directories if they don't exist
        if self.uwContext.rank()==0:
            if not os.path.isdir(actualpath):
                os.makedirs(actualpath)


        self.uwContext.barrier()
        #now save the object with the
        for key, val in self.objDict.items():
            val.save(os.path.join(actualpath,key + suffix))

    def saveAll():
        """
        Save objDict and dictDict
        """
        return NotImplemented



    def cleanForward():
        """
        As checkpoint progesses we may want to delete previous checkpoint dirs
        """
        return NotImplemented

    def cleanBackward():
        """
        As checkpoint progesses we may want to delete previous checkpoint dirs
        """
        return NotImplemented
