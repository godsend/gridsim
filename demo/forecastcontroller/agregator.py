# -*- coding: utf-8 -*-
"""
.. codeauthor:: Joel Cavat <joel.cavat@hesge.ch>

This file implement the :class:`AgregatorSimulator` which is a module for gridsim. This represent a central
controller which manage a set of local controllers implemented by :class:`AgregatorElement`.

The central controller send information to the local controllers : the cost and the temperature forecast
in degree for each slot duration.

During the elapsing time, temperature could be different from the forecast. It leads the local controllers to
difference of value. The local controllers can detect such an errors and recompute the decision. The central controllers
could send new costs vector during the simulation, different from the forecast. In this case, the local controllers
need to recompute their decision, too.

The decision of an local controller is an on/off decision for each duration slot.

The :class:`AgregatorElement` is an intermediate abstract class to receive the new cost and detect this receipt to
recompute

"""

import sys
import os
sys.path.append(os.path.abspath("../"))

from gridsim.simulation import Simulator, AbstractSimulationElement, Position
from gridsim.controller import ControllerSimulator, AbstractControllerElement
from gridsim.thermal.element import TimeSeriesThermalProcess
from gridsim.thermal.core import ThermalProcess, ThermalCoupling
from gridsim.unit import units

# Library used for making linear problems
from pulp import *
import random

# Used for statistics
import numpy

# Library used for evolutionary algorithms
from deap import base
from deap import creator
from deap import tools
from deap import algorithms


class AgregatorSimulator(ControllerSimulator):
    """
    This represent a central controller which manage a set of local controllers
    implemented by :class:`AgregatorElement`.

    The central controller send information to the local controllers : the cost and the temperature forecast
    in degree for each slot duration.
    """

    def __init__(self):
        """
        Simulation module constructor
        """
        super(AgregatorSimulator, self).__init__()
        self._decision_time = 0

        self.outside_process = None
        self.outside_temperature = {}
        self.temperature = 0
        self.cost_reference = [0]
        """
        The cost reference for a day type.
        """

    @property
    def decision_time(self):
        """
        Getter for the decision time
        :return: the decision time
        """
        return self._decision_time

    @decision_time.setter
    def decision_time(self, value):
        """
        Setter for the decision time
        :param value: the decision time
        """
        self._decision_time = units.value(units.convert(value, units.second))

    def attribute_name(self):
        """
        Refer to the module name
        :return: The module name
        """
        return 'agregator'

    def add(self, element):
        """
        Adds the control element to the controller simulation module.
        :param element: Element to add to the control simulator module.
        """
        if isinstance(element, AgregatorElement):
            element.id = len(self._controllers)
            self._controllers.append(element)
            return element
        return None


    def calculate(self, time, delta_time):
        """
        Method where the temperature forecast is computed and sent to the local controllers and where
        the cost is computed ans sent to the local controller, too
        :param time: The actual time reference
        :param delta_time: The delta time
        """

        common_unit = units.unit(units.convert(time, units.second))
        time = units.value(units.convert(time, units.second))
        delta_time = units.value(units.convert(delta_time, units.second))

        # If this is the decision time we simulate a cost and a forecast for the temperature.
        if int(time) % int(self._decision_time) == 0:

            j = 0
            cost = {}
            for i in range(int(time), int(time + self._decision_time), int(delta_time)):
                cost[i] = max(0, self.cost_reference[j] + random.normalvariate(0, 1))
                j += 1

            # Add the first cost for the day next
            # TODO: Improve procedure without this fix
            cost[int(time + self._decision_time)] = self.cost_reference[0]

            for controller in self._controllers:         
                # Push the cost & Compute
                controller.cost = cost
                controller.outside_temperature_forecast = self.__temperature(time, time + self._decision_time + delta_time, delta_time, common_unit)

            
        #
        # Rest of the time
        #
        for controller in self._controllers:  
            controller.calculate(time, delta_time)

    def __temperature(self, start, stop, delta_time, common_unit):

        if self.outside_process is not None:
            temperature_outside_forecast = {}
            for k in range(int(start), int(start+stop), int(delta_time)):
                self.outside_process.set_time(units(k, common_unit))
                corr = random.normalvariate(0, 0.2)
                # corr = 0
                # corr = 4
                # if k > (start+stop)/2:
                #     corr = -1

                temperature_outside_forecast[k] = units.value(units.convert(getattr(self.outside_process, "temperature"), units.celsius)) + corr
            return temperature_outside_forecast
        else:
            return self.outside_temperature


class AgregatorElement(AbstractControllerElement):
    def __init__(self, friendly_name, position=Position()):
        super(AgregatorElement, self).__init__(friendly_name)
        self._cost = {}
        self._cost_has_changed = False

    @property
    def cost(self):
        return self._cost
    @cost.setter
    def cost(self, value):
        # Merge the correction
        self._cost = dict(self._cost.items() + value.items())
        self._cost_has_changed = True

    def total_cost(self):
        raise NotImplementedError('Pure abstract method!')

    def total_power(self):
        raise NotImplementedError('Pure abstract method!')

    def reset(self):
        raise NotImplementedError('Pure abstract method!')

    def calculate(self, time, delta_time):
        raise NotImplementedError('Pure abstract method!')

    def update(self, time, delta_time):
        raise NotImplementedError('Pure abstract method!')


class ForecastController(AgregatorElement):

    class Correction(object):
        def __init__(self, thermal_conductivity, temperature):
            self.thermal_conductivity = thermal_conductivity
            self.temperature = temperature



    def __init__(self, friendly_name, target_temperature, hysteresis, thermal_process, subject, attribute,
                 decision_time, delta_time, on_value=True, off_value=False, position=Position()):

        """
        A forecast control.ler. This class optimizes a thermal process given to a weather forecast and
        a cost vector.

        :param: friendly_name: User friendly name to give to the element.
        :type friendly_name: str, unicode
        :param: target_temperature: The temperature to try to maintain inside the target ThermalProcess.
        :type: target_temperature: float, int
        :param: hysteresis: The +- hysteresis in order to keep in an area of variation of temperature
        :type: hysteresis: float, int
        :param: thermal_process: The reference to the thermal process to observe.
        :type: thermal_process: ThermalProcess
        :param: subject: Reference to the object of which's attribute has to be changed depending on the termperature.
        :type: object
        :param: attribute: The name of the attribute to control as string.
        :type: str, unicode
        :param: data_path: The file path for forecast and cost
        :type: str, unicode
        :param decision_time: Step of decision
        :type decision_time: float
        :param delta_time: Time interval for the simulation in seconds.
        :type delta_time: float
        :param: on_value: The value to set for the attribute in order to turn the device "on".
        :type: on_value: any
        :param: off_on_value: The value to set for the attribute in order to turn the device "off".
        :type: off_value: any
        :param position: The position of the thermal element. Defaults to [0,0,0].
        :type position: :class:`Position`
        """
        super(ForecastController, self).__init__(friendly_name, position)
        if not isinstance(target_temperature, units.Quantity):
            raise TypeError('target_temperature')
        self.target_temperature = units.value(target_temperature)
        """
        The temperature to try to retain inside the observer thermal process by conducting an electrothermal element.
        """

        if not isinstance(hysteresis, (float, int)):
            raise TypeError('hysteresis')
        self.hysteresis = hysteresis
        """
        The +- hysteresis applied to the temperature measure in order to avoid to fast on/off switching.
        """

        if not isinstance(thermal_process, AbstractSimulationElement) and not hasattr(thermal_process, 'temperature'):
            raise TypeError('thermal_process')
        self.thermal_process = thermal_process
        """
        The reference to the thermal process to observe and read the temperature from.
        """

        if not isinstance(subject, AbstractSimulationElement):
            raise TypeError('subject')
        self.subject = subject
        """
        The reference to the element to control.
        """

        if not isinstance(attribute, (str, unicode)):
            raise TypeError('attribute')
        self.attribute = attribute
        """
        Name of the attribute to control.
        """

        self.on_value = on_value
        """
        Value to set in order to turn the element on.
        """

        self.off_value = off_value
        """
        Value to set in order to turn the element off.
        """

        self.decision_time = int(units.value(units.convert(decision_time, units.second)))
        """
        Value of decision time. The optimization will be computed each step
        """

        self.delta_time = int(units.value(units.convert(delta_time, units.second)))
        """
        Value of the unit of time used to decide the optimized comsumption
        """



        self.error = 0
        self.count_error = 0
        self.absolute_error = 0
        self.total_error = 0
        self.total_absolute_error = 0
        self.countOptimization = 0
        self.mean = target_temperature

        self._outside_temperature_forecast = {}

        

        self._instant_cost = 0 # for plotting
        self._total_cost = 0
        self._total_power = 0
        self._output_value = off_value
        self._power_on = {0: 0}
        self._history_temperature = []
        self._historic_for_correction = []
        self._current_correction = None # Correction if room next
        self._current_temperature_correction = 0.
        self._outside_process = None
        self._outside_coupling = None
        self._external_process = [] # not used yet
        self._external_coupling = [] # not used yet  

        ForecastController.MAX_TOTAL_ABSOLUTE_ERROR  = 1./self.delta_time
        ForecastController.MAX_TOTAL_ERROR = 0.10
        ForecastController.MAX_REALTIME_ERROR  = 0.5
        ForecastController.MAX_DAY_HISTORIC = 10
        ForecastController.NEXT_STEP_MIN = 10

    @property
    def outside_temperature_forecast(self):
        return self._outside_temperature_forecast
    @outside_temperature_forecast.setter
    def outside_temperature_forecast(self, value):
        self._current_temperature_correction = 0.
        self._outside_temperature_forecast = units.value(value)


    # AbstractSimulationElement implementation.
    def reset(self):
        """
        AbstractSimulationElement implementation, see :func:`agreflex.core.AbstractSimulationElement.reset`.
        """
        pass

    def __optimize(self, cost, first_decision = 0, correction = None, delta_temperature_outside_correction = 0.):

        ###########################################################
        #
        # Set the datas
        #

        starting_time = sorted(cost.keys())[0]
        ending_time = sorted(cost.keys())[-1] + self.delta_time

        # Value importation
        # external_thermal_element = self._outside_process # Get the temperature at time t
        thermal_conductivity = units.value(getattr(self._outside_coupling, "thermal_conductivity")) # with couple
        thermal_capacity = units.value(self.thermal_process.thermal_volumic_capacity)
        if thermal_capacity == 0:
            raise RuntimeError("Thermal capacity must be greater than zero")

        subject_energy = units.value(self.subject.power)
        subject_efficiency = getattr(self.subject, "_efficiency_factor")

        nb_slots_optimization = len(cost)


        #
        # Set the datas
        #
        ###########################################################


        ###########################################################
        #
        # Prepare the problem
        #

        # Linearization of the problem
        problem = LpProblem("lpOptimization", LpMinimize)
        # Decisions variables
        initial_temperature = {}
        final_temperature = {}
        _power_on = {}
        exceeding = {}
        difference_mean_temperature = LpVariable("diff_mean", lowBound = 0)

        # Datas
        temperature = {}
        solar_radiation = {}

        # For each step of optimisation
        for t in range(starting_time, ending_time, self.delta_time):
            # Prepare the variables
            initial_temperature[t] = LpVariable("tinit_%s" % t)
            final_temperature[t] = LpVariable("tfin_%s" % t)
            _power_on[t] = LpVariable("p_%s" % t, cat='Binary')
            exceeding[t] = LpVariable("exc_%s" % t, lowBound=0)
            temperature[t] = units.value(units.convert(self._outside_temperature_forecast[t], units.celsius)) + delta_temperature_outside_correction


        # Objective to minimize the cost
        problem += lpSum(_power_on[t] * cost[t] + 1000 * exceeding[t] for t in cost.keys()) + 10000 * difference_mean_temperature


        #
        # Constraints
        #

        # Temperature mean during the time optimization must be the target temperature (greater than because lost precision float)
        problem += lpSum(final_temperature[t] for t in final_temperature.keys()) >= units.value(units.convert(self.target_temperature, units.celsius)) * (nb_slots_optimization) - difference_mean_temperature


        if correction == None:
            for t in range(starting_time, ending_time, self.delta_time):

                outside_leverage = self.__external_leverage(\
                    thermal_capacity = thermal_capacity, \
                    thermal_conductivity = thermal_conductivity, \
                    external_temperature = temperature[t], \
                    internal_temperature = initial_temperature[t])

                inner_production = self.__internal_production( \
                    internal_temperature = initial_temperature[t], \
                    power_on = _power_on[t], \
                    subject_efficiency = subject_efficiency, \
                    subject_energy = subject_energy, \
                    thermal_capacity = thermal_capacity)

                problem += final_temperature[t] == outside_leverage + inner_production
                
        else:
            for t in range(starting_time, ending_time, self.delta_time):
                outside_leverage = self.__external_leverage( \
                    thermal_capacity = thermal_capacity, \
                    thermal_conductivity = thermal_conductivity,\
                    external_temperature = temperature[t], \
                    internal_temperature = initial_temperature[t])

                inner_production = self.__internal_production( \
                    internal_temperature = initial_temperature[t], \
                    power_on = _power_on[t], \
                    subject_efficiency = subject_efficiency, \
                    subject_energy = subject_energy, \
                    thermal_capacity = thermal_capacity)

                correction_leverage = self.__external_leverage( \
                    thermal_capacity = thermal_capacity, \
                    thermal_conductivity = correction.thermal_conductivity, \
                    external_temperature = correction.temperature, \
                    internal_temperature = initial_temperature[t])
                
                problem += final_temperature[t] == outside_leverage + inner_production + correction_leverage

        for t in range(starting_time, ending_time, self.delta_time):
            problem += final_temperature[t] <= self.target_temperature + 0.5 * self.hysteresis + exceeding[t]
            problem += final_temperature[t] >= self.target_temperature - 0.5 * self.hysteresis - exceeding[t]
        problem += initial_temperature[starting_time] == units.value(units.convert(self.thermal_process.temperature, units.celsius))
        
        problem += _power_on[starting_time] == first_decision

        for t in range(starting_time + self.delta_time, ending_time, self.delta_time):
            problem += initial_temperature[t] == final_temperature[t-self.delta_time]


        #
        # Prepare the problem 
        #
        ###########################################################


        ###########################################################
        #
        # Resolution and return
        #

        # status = problem.solve(COIN_CMD("/opt/coin-Cbc-2.8/bin/cbc"))
        self.status = problem.solve(GUROBI(msg=0))
        self.status = LpStatus[problem.status]
        if self.status is 'Infeasible':
            raise RuntimeError("The problem is'nt feasible")

        return [dict([(k,value(v)) for k,v in _power_on.items()]), dict([(k, value(v)) for k,v in final_temperature.items()])]

    def __is_delta_temperature_error(self):


        ######################################################################################################################
        #
        # Load & prepare datas
        #

        subject_efficiency = getattr(self.subject, "_efficiency_factor")
        subject_thermal_capacity = units.value(self.thermal_process.thermal_volumic_capacity)
        subject_thermal_outside_coupling = units.value(self._outside_coupling.thermal_conductivity)


        temperature = []
        temperature.append(self._historic_for_correction[0][1])
        error = 0.

        #
        # Load & prepare datas
        #
        ######################################################################################################################


        ######################################################################################################################
        #
        # Compute difference of temperature regarding to the real temperature in contrast to the forecast temperature
        #
        for time, init_temp, opt_temp, found_temp, temperature_ext_forecast, temperature_ext, power in self._historic_for_correction:

            last_temperature = temperature[-1]

            outside_leverage = self.__external_leverage(\
                thermal_capacity = subject_thermal_capacity, \
                thermal_conductivity = subject_thermal_outside_coupling, \
                external_temperature = temperature_ext, \
                internal_temperature = last_temperature)

            inner_production = self.__internal_production(\
                internal_temperature = init_temp, \
                power_on = power, \
                subject_efficiency = subject_efficiency, \
                subject_energy = units.value(self.subject.power), \
                thermal_capacity = subject_thermal_capacity)

            correction_leverage = 0
            if self._current_correction is not None:
                correction_leverage = self.__external_leverage(\
                    thermal_capacity = subject_thermal_capacity, \
                    thermal_conductivity = self._current_correction.thermal_conductivity, \
                    external_temperature = units.value(self._current_correction.temperature), \
                    internal_temperature = last_temperature)

            final_temperature = outside_leverage + inner_production + correction_leverage

            error += found_temp - final_temperature
             
            temperature.append(final_temperature)

        # TODO Constant for 0.05
        print "| | +", abs(error), 0.05 * len(self._historic_for_correction)
        return abs(error) <= 0.05 * len(self._historic_for_correction)

        #
        # Compute difference of temperature regarding to the real temperature in contrast to the forecast temperature
        #
        ######################################################################################################################



    def __check_correction(self):

        print "| | + Check correction", len(self._historic_for_correction)
        if len(self._historic_for_correction) == 0:
            return self._current_correction, False

        #
        # Controle if the error is due to difference between real and forecasted temperature
        #
        if self.__is_delta_temperature_error():
            print "| | | + Temperature error"
            return self._current_correction, False



        ######################################################################################################################
        #
        # Load datas & prepare genetic components
        #

        subject_efficiency = getattr(self.subject, "_efficiency_factor")
        subject_thermal_capacity = units.value(self.thermal_process.thermal_capacity)
        subject_thermal_outside_coupling = units.value(self._outside_coupling.thermal_conductivity)

        # Create individual with his attributes   
        creator.create("FitnessMinError", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMinError)

        toolbox = base.Toolbox()

        # Attribute generator
        toolbox.register("thermal_conductivity", random.randint, -1000., 1000.)
        toolbox.register("temperature", random.randint, -50, 50)

        # Structure initializers
        toolbox.register("individual", tools.initCycle, creator.Individual, \
            (toolbox.thermal_conductivity, toolbox.temperature), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        #
        # Load datas & prepare genetic components
        #
        ######################################################################################################################


        ######################################################################################################################
        #
        # Fitness evaluation & mutation operator
        #
        def evaluate(individual):

            weight = 0.0


            individual_thermal_conductivity = individual[0]
            individual_constant_temperature = individual[1]


            temperature = []
            temperature.append(self._historic_for_correction[0][1])

            for time, init_temp, opt_temp, found_temp, temperature_ext_forecast, temperature_ext, power in self._historic_for_correction:

                last_temperature = temperature[-1]


                outside_leverage = self.__external_leverage(\
                    thermal_capacity = subject_thermal_capacity, \
                    thermal_conductivity = subject_thermal_outside_coupling, \
                    external_temperature = temperature_ext + self._current_temperature_correction, \
                    internal_temperature = last_temperature)

                inner_production = self.__internal_production(\
                    internal_temperature = init_temp, \
                    power_on = power, \
                    subject_efficiency = subject_efficiency, \
                    subject_energy = units.value(self.subject.power), \
                    thermal_capacity = subject_thermal_capacity)

                if self._current_correction== None:
                    correction_leverage = self.__external_leverage(\
                        thermal_capacity = subject_thermal_capacity,\
                        thermal_conductivity = individual_thermal_conductivity, \
                        external_temperature = individual_constant_temperature, \
                        internal_temperature = last_temperature)
                else:
                    correction_leverage = self.__external_leverage(\
                        thermal_capacity = subject_thermal_capacity, \
                        thermal_conductivity = individual_thermal_conductivity+self._current_correction.thermal_conductivity, \
                        external_temperature = individual_constant_temperature+self._current_correction.temperature, \
                        internal_temperature = last_temperature)

                final_temperature = outside_leverage + inner_production + correction_leverage

                temperature.append(final_temperature)

                weight += abs(found_temp - final_temperature) * 10000

            return weight,


        def mutSet(individual):

            individual[0] += random.uniform(-10,10) # thermal_coupling
            individual[1] += random.uniform(-5,5) # temperature

            return individual,

        #
        # Fitness evaluation & mutation operator
        #
        ######################################################################################################################




        ######################################################################################################################
        #
        # Initialize & compute
        #


        toolbox.register("evaluate", evaluate)
        # toolbox.register("mate", cxSet)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        # toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mutate", mutSet)
        toolbox.register("select", tools.selTournament, tournsize=3)
        # toolbox.register("select", tools.selNSGA2)

        NGEN = 60 # 60  # 200
        LAMBDA = 300 #300 # 500
        CXPB = 0.3 #0.5
        MUTPB = 0.4 #0.6

        pop = toolbox.population(n=LAMBDA)
        hof = tools.ParetoFront()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean, axis=0)
        stats.register("std", numpy.std, axis=0)
        stats.register("min", numpy.min, axis=0)
        stats.register("max", numpy.max, axis=0)

        pop, log = algorithms.eaSimple(pop, toolbox, stats=stats, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, halloffame=hof, verbose=False)
        best = tools.selBest(pop, 1)[0]
        score = best.fitness.values[0]


        #
        # Initialize & compute
        #
        ######################################################################################################################


        ######################################################################################################################
        #
        # Compute the actual score and compare it with the new
        # If the new is better, we swap the current correction
        #
        temperature = []
        temperature.append(self._historic_for_correction[0][1])
        old_score = 0.   
        score_basic = 0.  
        for time, init_temp, opt_temp, found_temp, temperature_ext_forecast, temperature_ext, power in self._historic_for_correction:
            old_score += abs(opt_temp - found_temp) * 10000

            last_temperature = temperature[-1]
            outside_leverage = self.__external_leverage(\
                thermal_capacity = subject_thermal_capacity, \
                thermal_conductivity = subject_thermal_outside_coupling, \
                external_temperature = temperature_ext, \
                internal_temperature = last_temperature)

            inner_production = self.__internal_production(\
                internal_temperature = init_temp, \
                power_on = power, \
                subject_efficiency = subject_efficiency, \
                subject_energy = units.value(self.subject.power), \
                thermal_capacity = subject_thermal_capacity)
            final_temperature = outside_leverage + inner_production
            temperature.append(final_temperature)
            score_basic += abs(found_temp - final_temperature) * 10000


        if score_basic < old_score and score_basic < score:
            print "| | | + reinit"            
            return None, True

        if score > old_score or score == float('nan'):
            print "| | | + keep current correction"
            return self._current_correction, False


        print "| | | + swap with correction:", best, "and", best.fitness.values[0],"fitness value [", 
        if self._current_correction== None:
            c = self.Correction(best[0], best[1])

            print c.thermal_conductivity, c.temperature, "]"
            return c, True


        c = self.Correction(best[0] + self._current_correction.thermal_conductivity, best[1] + self._current_correction.temperature)
        print c.thermal_conductivity, c.temperature,"]"
        return c, True

        #
        # Compute the actual score and compare it with the new
        #
        ######################################################################################################################

    def calculate(self, time, delta_time):
        """
        AbstractSimulationElement implementation, see :func:`agreflex.core.AbstractSimulationElement.calculate`.
        """


        #
        # Update the cost vector regarding to the time
        #
        self._cost = {k: v for k,v in self._cost.items() if k >= time}



        ######################################################################################################################
        #
        # Compute decision for each delta_time 
        # If decision time or if cost has changed
        # We can't update the actual state. So we take decision for the next step.
        #

        if self._cost_has_changed:

            self._cost_has_changed = False

            print "+ Decision time", int(time / self.decision_time)

            # Maximum time for the current or future optimization
            ForecastController.target_time = int(time + self.decision_time + 1 * self.delta_time)

            nb_slots_optimization = len(self._cost)

            # Conditions to minimize errors correction
            if self.count_error > 0:
                admissible_error = self.total_absolute_error / self.count_error

                #
                # Absolute error is too big
                # 
                if  admissible_error > ForecastController.MAX_TOTAL_ABSOLUTE_ERROR:
                    print "| + absolute error override"
                    self._current_correction, fountBetterCorrection = self.__check_correction()
                    if fountBetterCorrection:
                        self._historic_for_correction = []

                #
                # Precision more precise but only with a full historic
                #
                if admissible_error > ForecastController.MAX_TOTAL_ABSOLUTE_ERROR / 10 and \
                    len(self._historic_for_correction) > (ForecastController.MAX_DAY_HISTORIC * nb_slots_optimization):
                    print "| + full historic"
                    self._current_correction, fountBetterCorrection = self.__check_correction()
                    self._historic_for_correction = []

                self.total_absolute_error = 0.0
                self.total_error = 0.0
                self.count_error = 0

            # Optimize
            if int(time) not in self._power_on.keys():
                self._power_on[int(time)] = 0

            self._power_on, self.temperature_optimal = self.__optimize( \
                cost = self._cost, \
                first_decision = self._power_on[int(time)], \
                correction = self._current_correction)

            print "| + classic optimized"

            #
            # Statistics and historic
            #
            self.countOptimization += 1
            if len(self._history_temperature) > 0:
                self._history_temperature = self._history_temperature[-96:]
                self.mean = numpy.mean(self._history_temperature)
                print "| | + Mean:",self.mean


        #
        # Compute decision for each delta_time or if cost has changed
        #
        ######################################################################################################################



        ######################################################################################################################
        #
        # Recompute in realtime ?
        # Check if we need to recompute the optimization due to a difference of forecast
        # Compare forecast temperature with real temperature and recompute if the difference is to high. Recompute if sufficient of historic
        if time - self.delta_time in self.temperature_optimal.keys():

            #
            # Compute the error
            # difference between optimal temperature computed and real temperature
            #
            self.error = self.temperature_optimal[time - self.delta_time] - units.value(units.convert(self.thermal_process.temperature, units.celsius))
            self.error = self.error if abs(self.error) >= 1e-10 else 0.0
            self.absolute_error = abs(self.error)
            self.total_error += self.error
            self.total_absolute_error += abs(self.error)
            self.count_error += 1


            
            #
            # Historic
            #
            self._outside_process.set_time(time * units.second)
            self._historic_for_correction.append((\
                time, \
                self.old_temperature, \
                self.temperature_optimal[time - self.delta_time], \
                units.value(units.convert(self.thermal_process.temperature, units.celsius)), \
                self._outside_temperature_forecast[time], \
                units.value(units.convert(getattr(self._outside_process, "temperature"), units.celsius)), \
                self._power_on[time - self.delta_time]))

            # print "| >", time, ":", self.absolute_error, self.total_error,  self.total_error / self.count_error, self.absolute_error > 2*abs(self.total_error / self.count_error)

            if self.absolute_error >= ForecastController.MAX_REALTIME_ERROR and self.absolute_error > 2 * abs(self.total_error / self.count_error):

                print "| + current error too big", self.error
                if self.__is_delta_temperature_error():

                    #
                    # Recompute according to the difference of temperature
                    #
                    self.__recompute_with_temperature_error(time)
                    self._historic_for_correction = []


                else:

                    # 
                    # Error due to a human behavior, must recompute
                    #
                    self._power_on, self.temperature_optimal = self.__optimize(\
                        cost = self._cost, \
                        first_decision = self._power_on[int(time)], \
                        correction= self._current_correction)

                    self.countOptimization += 1
                    self._historic_for_correction = []


                self.total_absolute_error = 0.0
                self.total_error = 0.0
                self.count_error = 0


            elif abs(self.total_error) / self.count_error >= ForecastController.MAX_TOTAL_ERROR and len(self._historic_for_correction) >= ForecastController.NEXT_STEP_MIN:
                
                print "| + total error too big", self.total_error

                if self.__is_delta_temperature_error():
                    
                    #
                    # Recompute according to the difference of temperature
                    #
                    self.__recompute_with_temperature_error(time)
                    self._historic_for_correction = []

                else:
                    #
                    # Compute correction due to adjacency room
                    #
                    self._current_correction, fountBetterCorrection = self.__check_correction()
                    if fountBetterCorrection:
                        #
                        # Reoptimize with corrections
                        #
                        self._historic_for_correction = []
                        self._power_on, self.temperature_optimal = self.__optimize(\
                            cost = self._cost, \
                            correction= self._current_correction)
                        self.countOptimization += 1

                self.total_absolute_error = 0.0
                self.total_error = 0.0
                self.count_error = 0
                print ""



        #
        # Recompute in realtime ?
        ######################################################################################################################


        #
        # Historic & Stats
        #
        self.old_temperature = units.value(units.convert(self.thermal_process.temperature, units.celsius))
        self._history_temperature.append(self.old_temperature)
        self._instant_cost = self._cost[time]
        self._total_cost += self._cost[time] * self._power_on[time]
        self._total_power += self._power_on[time] * self.subject.power

        #
        # Decision for the next step
        #
        self._output_value = self.on_value if int(self._power_on[time + delta_time]) == 1 else self.off_value


    def update(self, time, delta_time):
        """
        AbstractSimulationElement implementation, see :func:`agreflex.core.AbstractSimulationElement.update`.
        """
        setattr(self.subject, self.attribute, self._output_value)

    def add(self, thermalProcess, thermalCoupling):
        """
        TimeSeriesThermalProcess implemented. Not other ThermalProcess impolemented yet
        TODO: Implement ConstantTemperatureProcess, ThermalProcess, process
        """

        if not isinstance(thermalCoupling, ThermalCoupling) or thermalCoupling is None:
            raise RuntimeError('Missing or invalid thermalCoupling reference.')

        if thermalProcess is not None:
            if isinstance(thermalProcess, TimeSeriesThermalProcess):
                self._outside_process = thermalProcess
                self._outside_coupling = thermalCoupling
                return
            if isinstance(thermalProcess, ThermalProcess):
                self._external_process.append(thermalProcess)
                self._external_coupling.append(thermalCoupling)
                return

        raise RuntimeError('Missing or invalid thermalProcess or thermalCoupling reference.')

    def total_cost(self):
        return self._total_cost
    def total_power(self):
        return self._total_power

    def __external_leverage(self, thermal_capacity, thermal_conductivity, external_temperature, internal_temperature):
        return (self.delta_time / thermal_capacity) * (thermal_conductivity * (external_temperature - internal_temperature))

    def __internal_production(self, internal_temperature, power_on, subject_efficiency, subject_energy, thermal_capacity):
        return internal_temperature + power_on * subject_efficiency * ((subject_energy / thermal_capacity) * self.delta_time)

    def __recompute_with_temperature_error(self, time):

        print "| | + due to temperature error", 
        if len(self._history_temperature) > 0:
            print numpy.mean(self._history_temperature[-20:])

        self._current_temperature_correction = units.value(units.convert(getattr(self._outside_process, "temperature"), units.celsius)) - self._outside_temperature_forecast[time]
        self._power_on, self.temperature_optimal = self.__optimize( \
            cost = self._cost, \
            first_decision = self._power_on[time], \
            correction = self._current_correction, \
            delta_temperature_outside_correction = self._current_temperature_correction)  

        print "| | |- Correction temperature", units.value(units.convert(getattr(self._outside_process, "temperature"), units.celsius)) - self._outside_temperature_forecast[time], units.value(units.convert(getattr(self._outside_process, "temperature"), units.celsius)) ,self._outside_temperature_forecast[time] 


Simulator.register_simulation_module(AgregatorSimulator)
