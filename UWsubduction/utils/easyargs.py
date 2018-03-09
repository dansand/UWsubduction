import pint

def easy_args(sysArgs, _dict):
    """
    Command line args are parsed and values go into the dictionary provided, under key names provided.
    Names are provided in the easyDict format, i.e. dictName.foo, rather than dictName['foo']
    This function will perform different updates, depending on the operator provided:

    * assign new value: =
    * in place multiply: *=
    * in place divide: /=
    * in place add: +=
    * in place subtract: -=



    Parameter
    ---------
    sysArgs : list
        the list of command line args provide by sys.argv (through the sys module)
    _dict:
        a dictionary to update values within

    Notes
    ---------
    ** This function *works* but is waiting on a rewrite
    There is currently no checking of the dictionary name (python object)
    with the name of dict in the command line arg. One could do:
        dp.foo=6
        foo.foo=7
    When easy_args(sysArgs, dp) is run, dp.foo will get upated.

    At the moment the main try/except block is what prevents crashing when a commandline arg is provided
    Which does not have a corresponding key. Bad.

    Examples
    ---------

    >>>python test.py dp.arg3+=24. dp.arg2='Cheryl', dp.arg4=True
    ...


    ...The above code would
    :add 24 to dictionary dp, key 'arg3',
    :replace dictionary dp, key 'arg2', with the string Cheryl
    :replace dictionary dp, key 'arg4', with the Boolean True


    """


    for farg in sysArgs:
        #Try to weed out some meaningless args.

        #print(farg)

        if ".py" in farg:
            continue
        if "=" not in farg:
            continue

        #print('cycle')
        try:
        #if 1:

            #########
            #Split out the dict name, key name, and value
            #########

            (dicitem,val) = farg.split("=") #Split on equals operator
            (dic,arg) = dicitem.split(".")
            if '*=' in farg:
                (dicitem,val) = farg.split("*=") #If in-place multiplication, split on '*='
                (dic,arg) = dicitem.split(".")
            if '/=' in farg:
                (dicitem,val) = farg.split("/=") #If in-place division, split on '/='
                (dic,arg) = dicitem.split(".")

            if '+=' in farg:
                (dicitem,val) = farg.split("+=") #If in-place addition, split on '+='
                (dic,arg) = dicitem.split(".")
            if '-=' in farg:
                (dicitem,val) = farg.split("-=") #If in-place addition, split on '-='
                (dic,arg) = dicitem.split(".")


            print('check dic,arg', dic,arg)


            #########
            #Basic type conversion from string to float, boolean
            #########

            if val == 'True':
                print('1')
                val = True
            elif val == 'False':     #First check if args are boolean
                print('2')
                val = False
            else:
                print('3')
                try:
                    val = float(val) #next try to convert  to a float,
                except ValueError:
                    pass             #otherwise leave as string

            #print('test', val,val_)

            #########
            #Resolve the units (using scaling/Pint)
            #########


            #if Pint quantity, we need to create a dimensional version for reassign, add/ subtract
            if hasattr(_dict[arg], 'dimensionality'):
                #need to catch ints here
                val_ = float(val)* 1.*_dict[arg].units

            else:
                val_ = val

            #########
            #Update the given dictionary
            #########
            #in place multiply/divides use val (the dimensional quant)
            #addition, subtraction, ressign use val_ which may have units

            try:
                if '*=' in farg:
                        _dict[arg] = _dict[arg]*val #multiply parameter by given factor
                elif '/=' in farg:
                        _dict[arg] = _dict[arg]/float(val) #divide parameter by given value
                elif '+=' in farg:
                        _dict[arg] = _dict[arg]+val_ #add to parameter given value
                elif '-=' in farg:
                        _dict[arg] = _dict[arg]-val_ #subtract from parameter given value
                else:
                        _dict[arg] = val_    #or reassign parameter by given value

            except:
                    pass

        except:
            pass
