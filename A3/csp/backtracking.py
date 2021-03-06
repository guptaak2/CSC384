from csp import Constraint, Variable, CSP
import random
import util

class UnassignedVars:
    '''class for holding the unassigned variables of a CSP. We can extract
       from, re-initialize it, and return variables to it.  Object is
       initialized by passing a select_criteria (to determine the
       order variables are extracted) and the CSP object.

       select_criteria = ['random', 'fixed', 'mrv'] with
       'random' == select a random unassigned variable
       'fixed'  == follow the ordering of the CSP variables (i.e.,
                   csp.variables()[0] before csp.variables()[1]
       'mrv'    == select the variable with minimum values in its current domain
                   break ties by the ordering in the CSP variables.
    '''
    def __init__(self, select_criteria, csp):
        if select_criteria not in ['random', 'fixed', 'mrv']:
            print "Error UnassignedVars given an illegal selection criteria {}. Must be one of 'random', 'stack', 'queue', or 'mrv'".format(select_criteria)
        self.unassigned = list(csp.variables())
        self.csp = csp
        self._select = select_criteria
        if select_criteria == 'fixed':
            #reverse unassigned list so that we can add and extract from the back
            self.unassigned.reverse()

    def extract(self):
        if not self.unassigned:
            print "Warning, extracting from empty unassigned list"
            return None
        if self._select == 'random':
            i = random.randint(0,len(self.unassigned)-1)
            nxtvar = self.unassigned[i]
            self.unassigned[i] = self.unassigned[-1]
            self.unassigned.pop()
            return nxtvar
        if self._select == 'fixed':
            return self.unassigned.pop()
        if self._select == 'mrv':
            nxtvar = min(self.unassigned, key=lambda v: v.curDomainSize())
            self.unassigned.remove(nxtvar)
            return nxtvar

    def empty(self):
        return len(self.unassigned) == 0

    def insert(self, var):
        if not var in self.csp.variables():
            print "Error, trying to insert variable {} in unassigned that is not in the CSP problem".format(var.name())
        else:
            self.unassigned.append(var)

def bt_search(algo, csp, variableHeuristic, allSolutions, trace):
    '''Main interface routine for calling different forms of backtracking search
       algorithm is one of ['BT', 'FC', 'GAC']
       csp is a CSP object specifying the csp problem to solve
       variableHeuristic is one of ['random', 'fixed', 'mrv']
       allSolutions True or False. True means we want to find all solutions.
       trace True of False. True means turn on tracing of the algorithm

       bt_search returns a list of solutions. Each solution is itself a list
       of pairs (var, value). Where var is a Variable object, and value is
       a value from its domain.
    '''
    varHeuristics = ['random', 'fixed', 'mrv']
    algorithms = ['BT', 'FC', 'GAC']

    #statistics
    bt_search.nodesExplored = 0

    if variableHeuristic not in varHeuristics:
        print "Error. Unknown variable heursitics {}. Must be one of {}.".format(
            variableHeuristic, varHeuristics)
    if algo not in algorithms:
        print "Error. Unknown algorithm heursitics {}. Must be one of {}.".format(
            algo, algorithms)

    uv = UnassignedVars(variableHeuristic,csp)
    Variable.clearUndoDict()
    for v in csp.variables():
        v.reset()
    solutions = []
    if algo == 'BT':
         solutions = BT(uv, csp, allSolutions, trace)
    elif algo == 'FC':
        for cnstr in csp.constraints():
            if cnstr.arity() == 1:
                FCCheck(c, None, None)  #FC with unary constraints at the root
        solutions = FC(uv, csp, allSolutions, trace)
    elif algo == 'GAC':
        GacEnforce(csp.constraints(), csp, None, None) #GAC at the root
        solutions = GAC(uv, csp, allSolutions, trace)

    return solutions, bt_search.nodesExplored

def BT(unAssignedVars, csp, allSolutions, trace):
    '''Backtracking Search. unAssignedVars is the current set of
       unassigned variables.  csp is the csp problem, allSolutions is
       True if you want all solutionss trace if you want some tracing
       of variable assignments tried and constraints failed. Returns
       the set of solutions found.

      To handle finding 'allSolutions', at every stage we collect
      up the solutions returned by the recursive  calls, and
      then return a list of all of them.

      If we are only looking for one solution we stop trying
      further value of the variable currently being tried as
      soon as one of the recursive calls returns some solutions.
    '''

    if unAssignedVars.empty():
        if trace: print "{} Solution Found".format(csp.name())
        soln = []
        for v in csp.variables():
            soln.append((v, v.getValue()))
        return [soln]  #each call returns a list of solutions found
    bt_search.nodesExplored += 1
    solns = []         #so far we have no solutions recursive calls
    nxtvar = unAssignedVars.extract()
    if trace: print "==>Trying {}".format(nxtvar.name())
    for val in nxtvar.domain():
        if trace: print "==> {} = {}".format(nxtvar.name(), val)
        nxtvar.setValue(val)
        constraintsOK = True
        for cnstr in csp.constraintsOf(nxtvar):
            if cnstr.numUnassigned() == 0:
                if not cnstr.check():
                    constraintsOK = False
                    if trace: print "<==falsified constraint\n"
                    break
        if constraintsOK:
            new_solns = BT(unAssignedVars, csp, allSolutions, trace)
            if new_solns:
                solns.extend(new_solns)
            if len(solns) > 0 and not allSolutions:
                break #don't bother with other values of nxtvar
                      #as we found a soln.
    nxtvar.unAssign()
    unAssignedVars.insert(nxtvar)
    return solns

def FCCheck(cnstr, reasonVar, reasonVal):
    if cnstr.numUnassigned() != 1:
        print "Error FCCheck called on constraint {} with {} neq 1 unassigned vars".format(cnstr.name(), cnstr.numUnassignedVars)
    var = cnstr.unAssignedVars()[0]
    for val in var.curDomain():
        var.setValue(val)
        if not cnstr.check():
            var.pruneValue(val, reasonVar, reasonVal)
        var.unAssign()  #NOTE WE MUST UNDO TRIAL ASSIGNMENT
    if var.curDomainSize() == 0:
        return "DWO"
    return "OK"

def FC(unAssignedVars, csp, allSolutions, trace):
    '''Forward checking search.
       unAssignedVars is the current set of
       unassigned variables.  csp is the csp
       problem, allSolutions is True if you want all solutionsl trace
       if you want some tracing of variable assignments tried and
       constraints failed.

       RETURNS LIST OF ALL SOLUTIONS FOUND.

       Finding allSolutions is handled just as it was in BT.  Except
       that when we are not looking for all solutions and we stop
       early because one of the recursive calls found a solution we
       must make sure that we restore all pruned values before
       returning.
    '''
    #your implementation for Question 2 goes in this function body.
    #you must not change the function parameters.
    #Implementing handling of the trace parameter is optional
    #but it can be useful for debugging

    listOfAllSolutions = []
    
    #Following code follows algorithm from lecture 6, slide 66
    
    #no more unassigned variables
    if unAssignedVars.empty():
        for var in csp.variables():
            #add all vars to list
            listOfAllSolutions.append((var, var.getValue()))
        return [listOfAllSolutions]
    
    #select next variable to assign
    var = unAssignedVars.extract()
    for val in var.curDomain(): #current domain
        var.setValue(val)
        noDWO = True
        for constraint in csp.constraintsOf(var):
            if constraint.numUnassigned() == 1:
                #prune future variables
                if FCCheck(constraint, var, val) == "DWO":
                    noDWO = False
                    break
        if noDWO:
            #add the new solutions to the list
            listOfAllSolutions.extend(FC(unAssignedVars, csp, allSolutions, trace))
            if len(listOfAllSolutions) > 0 and not allSolutions:
                #We want to stop after finding first 
                var.restoreValues(var, val)
                break
        
        #restore values that were pruned
        var.restoreValues(var, val)
    
    #reset var and restore unAssignedVars to var
    var.setValue(None)
    unAssignedVars.insert(var)

    #Return the list of all solutions
    return listOfAllSolutions
    
def GacEnforce(constraints, csp, reasonVar, reasonVal):
    '''Establish GAC on constraints by pruning values
       from the current domains of the variables.
       Return "OK" if completed "DWO" if found
       a domain wipe out.'''
    #your implementation for Question 3 goes in this function body
    #you must not change the function parameters
    #ensure that you return one of "OK" or "DWO"

    #using the algorithm from lecture
    #Loop while constraints are not empty
    constraintsLen = len(constraints)
    while not constraintsLen == 0:
        #Make cnstr gac
        
        #Get a constraint
        #never fails since we already checked length
        constraint = constraints.pop(0)
        i=0
        while i < len(constraint.scope()):
            var = constraint.scope()[i]
            for domainVal in var.curDomain():
                if not constraint.hasSupport(var, domainVal):
                    #prune 
                    var.pruneValue(domainVal, reasonVar, reasonVal)
                    if var.curDomainSize() == 0:
                        #Domain wipe out
                        return "DWO"
                    #If not domain wipe out, then recheck
                    for recheckConstraint in csp.constraintsOf(var):
                        if not recheckConstraint in constraints and recheckConstraint != constraint:
                            #add this to the constraints collection
                            constraints.append(recheckConstraint)
            i+=1
        #Update the length
        constraintsLen = len(constraints)
    return "OK"

def GAC(unAssignedVars, csp, allSolutions, trace):
    '''GAC search.
       unAssignedVars is the current set of
       unassigned variables.  csp is the csp
       problem, allSolutions is True if you want all solutionsl trace
       if you want some tracing of variable assignments tried and
       constraints failed.

       RETURNS LIST OF ALL SOLUTIONS FOUND.

       Finding allSolutions is handled just as it was in BT.  Except
       that when we are not looking for all solutions and we stop
       early because one of the recursive calls found a solution we
       must make sure that we restore all pruned values before
       returning.
    '''
    #your implementation for Question 3 goes in this function body
    #You must not change the function parameters.
    #implementing support for "trace" is optional, but it might
    #help you in debugging

    listOfAllSolutions = []
    #using algorithm from lecture:
    if unAssignedVars.empty():
        for var in csp.variables():
            listOfAllSolutions.append((var, var.getValue()))
        return [listOfAllSolutions]

    #select next variable to be assigned
    unassignedVar = unAssignedVars.extract()
    for domainVal in unassignedVar.curDomain():
        unassignedVar.setValue(domainVal)
        noDWO = True
        gacEnforceValue = GacEnforce(csp.constraintsOf(unassignedVar), csp, unassignedVar, domainVal)
        if gacEnforceValue == "DWO":
            #only var's domain changed-constraints with var have to be checked
            noDWO = False
        else: #OK was returned
            if noDWO:
                listOfAllSolutions.extend(GAC(unAssignedVars, csp, allSolutions, trace))
                if len(listOfAllSolutions) > 0 and  not allSolutions:
                    unassignedVar.restoreValues(unassignedVar, domainVal)
                    break
        unassignedVar.restoreValues(unassignedVar, domainVal)

    #set var to be unassigned and return to list
    unassignedVar.setValue(None)
    unAssignedVars.insert(unassignedVar)

    #Return the list of all solutions
    return listOfAllSolutions
