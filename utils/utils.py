def get_value(s, curr, squeeze=False):
    if "[" in s:
        dict_name = s.split("[")[0]
        key = s.split("[")[1].split("]")[0]
        if squeeze:
            return getattr(curr, dict_name)[key].squeeze()
        else:
            return getattr(curr, dict_name)[key]
    else:
        if squeeze:
            return getattr(curr, s).squeeze()
        else:
            return getattr(curr, s)

def get_processed_value(entry, module_funcs, module_vars):
    #module_funcs is where to get the function from. eg, if that is 'eval', it will get the required function from eval.py
    #module_vars is where to get the varibales from. eg, if that is 'curr', it will get the required function from main.py

    if entry[0]=='': # just store var name
        var_value = get_value(entry[1], module_vars)
    else: #get value from function
        func = entry[0]
        f_args = entry[1]
        var_value = getattr(module_funcs, func)(*[get_value(a, module_vars) for a in f_args])
    return var_value


def accumulate(model1, model2, decay=0):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)