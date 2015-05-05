# -*- coding: utf-8 -*-
"""
.. codeauthor:: Joel Cavat <joel.cavat@hesge.ch>

This demo can be run from the demo folder entering "python forecastcontroller/forecastcontroller.py" inside
the console.

This demo use an agregator as a main controller which pilot some forecastcontroller attached to a
ElectroThemralHeaterCooler device.

The main controller send a forecast each decision time regarding the meteo and the cost for the next period.
Each local controller compute a forecast of on/off running decision for each step time. Then, during the simulation,
if the temperature change or if the cost change, the main controller can send again these information to each
local forecastcontroller. They can readjust their on/off running decision.


"""
import sys
import os
import random

sys.path.append(os.path.abspath("../"))

from gridsim.unit import units
from gridsim.simulation import Simulator
from gridsim.recorder import PlotRecorder
from gridsim.thermal.element import TimeSeriesThermalProcess
from gridsim.thermal.core import ThermalProcess, ThermalCoupling
from gridsim.electrical.core import AbstractElectricalCPSElement
from gridsim.electrical.network import ElectricalPQBus, \
    ElectricalTransmissionLine
from gridsim.electrical.loadflow import DirectLoadFlowCalculator
from gridsim.timeseries import SortedConstantStepTimeSeriesObject
from gridsim.iodata.input import CSVReader
from gridsim.iodata.output import FigureSaver

from agregator import ForecastController


class ElectroThermalHeaterCooler(AbstractElectricalCPSElement):
    """
    Class made by Michael Clausen used in the first version of Agreflex. This class wasn't used in gridsim anymore.
    I import it and use the same mathematical model to simulate heat pomp (heater, cooler,...).
    """
    def __init__(self, friendly_name, pwr, efficiency_factor, thermal_process):
        """
        Electrical heat or cool element. If the efficiency factor is more than 0 the element heats, if less than 0 the
        element cools down the associated thermal process. You can even simulate heat pumps by setting the efficiency
        factor to values greater than 1 or smaller than -1.

        :param: friendly_name: User friendly name to give to the element.
        :type friendly_name: str, unicode
        :param: power: Electrical power of the heating/cooling element.
        :type: float, int
        :param: efficiency_factor: The efficiency factor [], 1.0 means a heater with 100% efficiency and -1 means a
            cooler with 100% efficiency.
        :type: float, int
        :param: thermal_process: Reference to the thermal process where to put/take the energy in/out to heat/cool.
        :type: thermal_process: ThermalProcess
        """
        super(ElectroThermalHeaterCooler, self).__init__(friendly_name)

        if not isinstance(efficiency_factor, (float, int)):
            raise TypeError('efficiency_factor must be a float or int!')
        self._efficiency_factor = float(efficiency_factor)

        if not isinstance(thermal_process, ThermalProcess):
            raise TypeError('thermal_process must be of type ThermalProcess!')
        self._thermal_process = thermal_process

        self.power = pwr

        self._on = False
        """
        Controls the heater/cooler. If this is True, the heater/cooler is active
        and takes energy from the electrical
        network to actually heat or cool the thermal process associated.
        """

        self.history = {}
        """
        Keep a history of values taken by self.on
        """

    @property
    def on(self):
        """
        Getter for parameter on/off

        :return: The value of the parameter. True, device is running, False, device is down
        :rtype: Boolean
        """
        return self._on

    @on.setter
    def on(self, on_off):
        """
        Setter for parameter on/off
        :param on_off: define if the device must work or not
        """
        self._on = on_off

    # AbstractSimulationElement implementation.
    def reset(self):
        """
        AbstractSimulationElement implementation, see :func:`agreflex.core.AbstractSimulationElement.reset`.
        """
        super(ElectroThermalHeaterCooler, self).reset()
        self.on = False

    def calculate(self, time, delta_time):
        """
        AbstractSimulationElement implementation, see :func:`agreflex.core.AbstractSimulationElement.calculate`.
        """
        self._internal_delta_energy = self.power * delta_time
        if not self.on:
            self._internal_delta_energy = 0*units.joule

        self.history[time] = self.on

    def update(self, time, delta_time):
        """
        AbstractSimulationElement implementation, see :func:`agreflex.core.AbstractSimulationElement.update`.
        """
        super(ElectroThermalHeaterCooler, self).update(time, delta_time)
        self._thermal_process.add_energy(
            self._delta_energy * self._efficiency_factor)


# Create simulator.
sim = Simulator()
sim.electrical.load_flow_calculator = DirectLoadFlowCalculator()

####################################################################################
# Basic configuration :
#
START_TIME = 0
DURATION_TIME = 1 * units.day
DECISION_DURATION_STEP = 1 * units.day
PERIOD_STEP = 30 * units.minute

# Buildings coupling with outdoor
TEMPERATURE_INIT_REF = 20
COUPLING_OUTSIDE_REF = 20

# Nb buildings with devices and controllers
MIN_INDEX_NB_DEVICES = 1
MAX_INDEX_NB_DEVICES = 1
MAX_DAY_HISTORIC = 20  # historic for corrections

# Example of costs
COST = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0,
        5.0, 5.0, 7.0, 7.0, 10.0, 10.0, 10.0, 10.0, 8.0, 8.0,
        4.0, 4.0, 3.0, 3.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0,
        5.0, 5.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0]

# Outside thermal element
outside = sim.thermal.add(TimeSeriesThermalProcess('outside', SortedConstantStepTimeSeriesObject(CSVReader()),
                                                   './data/example_time_series.csv',
                                                   lambda t: t*units.hour,
                                                   temperature_calculator=
                                                   lambda t: units.convert(units(t, units.degC),
                                                                           units.kelvin)))

# This list is used to register and graph some information about temperature and consumptions
lstTemperatureMonitoring = [outside]

# Create a minimal electrical simulation network with a thermal heater connected to Bus0.
bus0 = sim.electrical.add(ElectricalPQBus('Bus0'))
sim.electrical.connect("Line0", sim.electrical.bus(0), bus0, ElectricalTransmissionLine('Line0',
                                                                                        1000*units.meter,
                                                                                        0.2*units.ohm))

# Lists used for the graph
lstForecast = []
lstHeater = []

# Creation of a set of building, their devices and their forecastcontroller and their own parameter
for i in range(MIN_INDEX_NB_DEVICES, MAX_INDEX_NB_DEVICES + 1):

    # Random parameters for buildings
    size = random.randint(500, 4000)
    power = random.randint(500, 4000)
    hysteresis = random.randint(1, 3)
    coupling = COUPLING_OUTSIDE_REF + random.randint(-15, 20)
    temperature = units(TEMPERATURE_INIT_REF + random.randint(-5, 10), units.degC)

    # Information about building parameters
    print "building %i" %i, ":"
    print "- size:", size
    print "- power:", power
    print "- hysteresis:", hysteresis
    print "- coupling:", coupling
    print "- temperature:", temperature

    building = sim.thermal.add(ThermalProcess.room('building_%s' % i,
                                                   size*units.meter*units.meter,
                                                   2.5*units.meter,
                                                   units.convert(temperature, units.kelvin)))

    # The controller need this method /!\ hack due to the new version of greedsim related to agreflex
    building.thermal_volumic_capacity = building.thermal_capacity * building._mass

    coupling_outside = sim.thermal.add(ThermalCoupling('building %i to outside' % i,
                                                       coupling*units.thermal_conductivity,
                                                       building,
                                                       outside))

    heater = sim.electrical.add(ElectroThermalHeaterCooler('heater %i' %i,
                                                           power*units.watt,
                                                           1.0,
                                                           building))
    sim.electrical.attach(bus0, heater)

    # Parameter for the Agregator
    sim.agregator.cost_reference = COST

    # Thermal heater with controller
    forecastController = sim.agregator.add(ForecastController('forecastController %i' % i,
                                                              temperature,
                                                              hysteresis,
                                                              building,
                                                              heater,
                                                              'on',  # default: switch on
                                                              DECISION_DURATION_STEP,  # decision step
                                                              PERIOD_STEP))
    forecastController.add(outside, coupling_outside)
    forecastController.max_day_historic = MAX_DAY_HISTORIC

    lstForecast += [forecastController]
    lstTemperatureMonitoring += [building]
    lstHeater += [heater]


sim.agregator.decision_time = DECISION_DURATION_STEP
sim.agregator.outside_process = outside


# Create a plot recorder that records the temperatures of all thermal processes.
temp = PlotRecorder('temperature')
sim.record(temp, lstTemperatureMonitoring, lambda context: units.convert(context.value, units.celsius))
 
# Create a plot recorder that records the power used by the electrical heater.
power = PlotRecorder('delta_energy')
sim.record(power, lstHeater,
           lambda context: context.value / context.delta_time)

# Create a plot recorder that records the cost
cost = PlotRecorder('_instant_cost')
sim.record(cost, [lstForecast[0]],
           lambda context: context.value)

error = PlotRecorder('error')
sim.record(error, lstForecast,
           lambda context: context.value)

mean = PlotRecorder('mean')
sim.record(mean, lstForecast,
           lambda context: context.value)


# Simulate
sim.reset()
sim.run(DURATION_TIME, PERIOD_STEP)

for forecastController in lstForecast:
    print "+-------------------------------"
    print "| - Nb optimisation:\t", forecastController.countOptimization
    print "| - Sigma error:\t", forecastController.total_error
    print "| - Sigma error abs:\t", forecastController.total_absolute_error
    print "| - Total power:\t", forecastController.total_power()
    print "| - Total cost:\t", forecastController.total_cost()

# Outputs
FigureSaver(temp, "Temperature").save('./forecastcontroller/output/temperature.png')
FigureSaver(power, "Power").save('./forecastcontroller/output/heater.png')
FigureSaver(cost, "Cost").save('./forecastcontroller/output/cost.png')
FigureSaver(error, "Erreur").save('./forecastcontroller/output/error.png')
FigureSaver(mean, "Moyenne").save('./forecastcontroller/output/mean.png')
