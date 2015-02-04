# -*- coding: utf-8 -*-
# Lancer l'exemple depuis ./demo :
# python forecastcontroller/forecastcontroller.py

import sys, os
sys.path.append(os.path.abspath("../"))

from gridsim.unit import units
from gridsim.util import Position
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
from gridsim.controller import AbstractControllerElement

# Import des outils utiles pour cet exemples
from agregator import ForecastController, AgregatorSimulator

import random

class ElectroThermalHeaterCooler(AbstractElectricalCPSElement):
    def __init__(self, friendly_name, pwr, efficiency_factor, thermal_process):

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

    @property
    def on(self):
        return self._on

    @on.setter
    def on(self, on_off):
        self._on = on_off

    # AbstractSimulationElement implementation.
    def reset(self):
        super(ElectroThermalHeaterCooler, self).reset()
        self.on = False

    def calculate(self, time, delta_time):
        self._internal_delta_energy = self.power * delta_time
        if not self.on:
            self._internal_delta_energy = 0*units.joule

    def update(self, time, delta_time):
        super(ElectroThermalHeaterCooler, self).update(time, delta_time)
        self._thermal_process.add_energy(
            self._delta_energy * self._efficiency_factor)



# Create simulator.
sim = Simulator()
sim.electrical.load_flow_calculator = DirectLoadFlowCalculator()

START = 0 #6 * units.month
DURATION = 2 * units.day
STEP = 30 * units.minute


####################################################################################
# Basic configuration :
#
# Building coupling with outdoor
TEMPERATURE_INIT = 20
COUPLING_OUTSIDE = 20

# Outside thermal element
outside = sim.thermal.add(TimeSeriesThermalProcess('outside', SortedConstantStepTimeSeriesObject(CSVReader()),
                                                   './data/example_time_series.csv',
                                                   lambda t: t*units.hour,
                                                   temperature_calculator=
                                                     lambda t: units.convert(units(t, units.degC),
                                                                             units.kelvin)))
lstTemperatureMonitoring = [outside]

# Create a minimal electrical simulation network with a thermal heater connected to Bus0.
bus0 = sim.electrical.add(ElectricalPQBus('Bus0'))
sim.electrical.connect("Line0", sim.electrical.bus(0), bus0, ElectricalTransmissionLine('Line0', 1000*units.meter, 0.2*units.ohm))

lstForecast = []
lstHeater = []

MIN = 1
MAX = 1#20
for i in range (MIN, MAX + 1):
      size = random.randint(500,4000) #600
      power = random.randint(500,4000)
      hysteresis = random.randint(1,4)
      coupling = COUPLING_OUTSIDE + random.randint(-15,20)
      temperature = units(TEMPERATURE_INIT + random.randint(-5, 10), units.degC)

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
      print "_____", building.thermal_capacity * building._mass
      coupling_outside = sim.thermal.add(ThermalCoupling('building %i to outside' % i, 
                                                         coupling*units.thermal_conductivity,
                                                         building,
                                                         outside))

      heater = sim.electrical.add(ElectroThermalHeaterCooler('heater %i' %i, 
                                                             power*units.kilowatt,
                                                             1.0,
                                                             building))
      sim.electrical.attach(bus0, heater)

      # Thermal heater with controller
      forecastController = sim.agregator.add(ForecastController('forecastController %i' % i, temperature, hysteresis, building, heater, 'on', 1*units.day, STEP))
      forecastController.add(outside, coupling_outside)
      forecastController.max_day_historic = 20

      # lstForecast += [forecastController]
      lstTemperatureMonitoring += [building]
      lstHeater += [heater]



sim.agregator.decision_time = 1*units.day
sim.agregator.outside_process = outside






####################################################################################
# Configuration with convergent rooms :
#
# convergent_room = sim.thermal.add(ThermalProcess.room('convergent_room', 500, 2.5, 15))
# coupling_room1 = sim.thermal.add(ThermalCoupling('coupling', 20, building, convergent_room)) 
# lstTemperatureMonitoring += [convergent_room]



####################################################################################
# Configuration with a constant room :
#
# constant_room_1 = sim.thermal.add(ConstantTemperatureProcess('constant_room_1', 24))
# coupling_room_constant = sim.thermal.add(ThermalCoupling('coupling_constant_room_1', 10, building, constant_room_1))
# lstTemperatureMonitoring += [constant_room_1]
# constant_room_2 = sim.thermal.add(ConstantTemperatureProcess('constant_room_2', 10))
# coupling_room_constant = sim.thermal.add(ThermalCoupling('coupling_constant_room_2', 1, building, constant_room_2))
# lstTemperatureMonitoring += [constant_room_2]



####################################################################################
# Add room with thermostat :
#
# thermostat_room = sim.thermal.add(ThermalProcess.room('thermostat_room', 500, 2.5, 30))
# coupling_thermostat_room = sim.thermal.add(ThermalCoupling('coupling_thermostat_room', 5, building, thermostat_room)) ####################################################################################################
# lstTemperatureMonitoring += [thermostat_room]

# bus1 = sim.electrical.add(ElectricalPQBus('Bus1'))
# sim.electrical.connect("Line1", sim.electrical.bus(1), bus1, ElectricalTransmissionLine('Line1', 1000, 0.2))
# heater2 = sim.electrical.add(ElectroThermalHeaterCooler('heater2', 1000, 1.0, thermostat_room))
# sim.electrical.attach(heater2, bus1)

# thermostat = sim.agregator.add(Thermostat('thermostat', 30.0, 0.5, thermostat_room, heater2, 'on'))
# # Or


####################################################################################
# Add room with forecast controller :
#
# TEMP = 30
# building2 = sim.thermal.add(ThermalProcess.room('building2', 500, 2.5, TEMP))
# coupling_2 = sim.thermal.add(ThermalCoupling('coupling_2', 20, building2, outside)) 
# coupling_with_building = sim.thermal.add(ThermalCoupling('coupling_with', 20, building2, building)) 
# lstTemperatureMonitoring += [building2]

# bus1 = sim.electrical.add(ElectricalPQBus('Bus1'))
# sim.electrical.connect("Line1", sim.electrical.bus(1), bus1, ElectricalTransmissionLine('Line1', 1000, 0.2))
# heater1 = sim.electrical.add(ElectroThermalHeaterCooler('heater1', 1000, 1.0, building2))
# # cooler1 = sim.electrical.add(ElectroThermalHeaterCooler('heater1', 1000, 1.0, building2))
# sim.electrical.attach(heater1, bus1)
# # sim.electrical.attach(cooler1, bus1)
# # thermostat = sim.agregator.add(Thermostat('thermostat', TEMP, 0.5, building2, heater1, 'on'))
# forecastController2 = sim.agregator.add(ForecastController('forecastController2', TEMP, 3.0, building2, heater1, 'on', cost_time_series, Simulator.DAY, STEP))
# forecastController2.add(outside, coupling_2)
# forecastController2.max_day_historic = 20
# lstForecast += [forecastController2]




# Create a plot recorder that records the temperatures of all thermal processes.
# temp = PlotRecorder('temperatures')
# sim.record(temp, lstTemperatureMonitoring)
# 
# # Create a plot recorder that records the power used by the electrical heater.
# power = PlotRecorder('delta_energy')
# sim.record(power, lstHeater,
#            lambda context: context.value / context.delta_time)
# 
# # Create a plot recorder that records the cost
# cost = PlotRecorder('_instant_cost')
# sim.record(cost, [lstForecast[0]],
#            lambda context: context.value)
# 
# error = PlotRecorder('Error')
# sim.record(error, lstForecast,
#            lambda context: context.value)
# 
# mean = PlotRecorder('Moyenne')
# sim.record(mean, lstForecast,
#            lambda context: context.value)



# Simulate
sim.reset()
#sim.time = START
sim.run(1*units.day, 30*units.minute)
#sim.run(DURATION, STEP)

for forecastController in lstForecast:
      print "+-------------------------------"
      print "| - Nb optimisation:\t", forecastController.countOptimization
      print "| - Sigma error:\t", forecastController.total_error
      print "| - Sigma error abs:\t", forecastController.total_absolute_error
      print "| - Total power:\t", forecastController.total_power()
      print "| - Total cost:\t", forecastController.total_cost()

# finalCost = 0
# listOn = [k for k,v in heater.history.items() if v]
# for k in listOn:
#   cost_time_series.set_time(k)
#   finalCost += getattr(cost_time_series, "cost_vector")


# print "Cout final avec forecastController:\t", finalCost, "; énergie totale:", len(listOn)


# Outputs
# FigureSaver(temp, "Temperature").save('./output/temperature.png')
# FigureSaver(power, "Power").save('./output/heater.png')
# FigureSaver(cost, "Cost").save('./output/cost.png')
# FigureSaver(error, "Erreur").save('./output/error.png')
# FigureSaver(mean, "Moyenne").save('./output/mean.png')




####################################################################################
# New simulation for comparison with a thermostat
# sim = Simulator()


# # Building coupling with outdoor
# sim.thermal.add(building)
# sim.thermal.add(outside)
# sim.thermal.add(coupling_outside)

# # Create a minimal electrical simulation network with a thermal heater connected to Bus0.
# sim.electrical.add(bus0)
# sim.electrical.connect("Line0", sim.electrical.bus(0), bus0, ElectricalTransmissionLine('Line0', 500, 0.2))
# sim.electrical.add(heater)
# sim.electrical.attach(heater, bus0)

# # Thermal heater with controller
# sim.thermal.add(cost_time_series)
# thermostat = sim.agregator.add(Thermostat('thermostat', 21.0, 3.0, building, heater, 'on'))

# # Simulate
# sim.reset()
# sim.run(DURATION, STEP)

# finalCostThermostat = 0
# listOn = [k for k,v in heater.history.items() if v]
# for k in listOn:
#   cost_time_series.set_time(k)
#   finalCostThermostat += getattr(cost_time_series, "cost_vector")


# print "Cout final avec forecastController:\t", finalCost, "; énergie totale:", len(listOn)

# print "Cout final avec thermostat:\t", finalCostThermostat, "gain de", (1. - 1./ finalCostThermostat * finalCost) * 100, "%", "; énergie totale:", len(listOn)
