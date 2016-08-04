#!/usr/bin/python
import math
import sys
import time

import numpy
import portalocker
import scipy.interpolate

import ephem
import oskar


def element_area_and_temperature(freq_hz):
    """Return element effective area and system temperauture at given frequency.

    Effective area and temperature values for SKA1-LOW provided
    by Eloy de Lera Acedo (eloy at mrao dot cam dot ac dot uk).

    Args:
        freq_hz (float): Frequency, in Hz.

    Returns:
        Tuple containing effective area, in m^2, and system temperature, in K.
    """
    # Element noise data.
    noise_data = {
        'freqs': [0.05e9, 0.07e9, 0.11e9, 0.17e9, 0.25e9, 0.35e9,
                  0.45e9, 0.55e9, 0.65e9],
        'a_eff': [1.8791, 1.8791, 1.8694, 1.3193, 0.6080, 0.2956,
                  0.2046, 0.1384, 0.0792],
        't_sys': [4.0409e3, 1.5029e3, 0.6676e3, 0.2936e3, 0.1402e3, 0.0873e3,
                  0.0689e3, 0.0607e3, 0.0613e3]
    }
    log_freq = numpy.log10(freq_hz)
    freqs = numpy.array(noise_data['freqs'])
    a_eff = numpy.array(noise_data['a_eff'])
    t_sys = numpy.array(noise_data['t_sys'])
    f_cut = 2

    # Interpolate to get effective area.
    if freq_hz <= freqs[f_cut]:
        f = scipy.interpolate.interp1d(numpy.log10(freqs[:f_cut+1]), 
            numpy.log10(a_eff[:f_cut+1]), kind='slinear')
        a_eff = 10**f(log_freq)
    else:
        f = scipy.interpolate.interp1d(numpy.log10(freqs[f_cut:]), 
            numpy.log10(a_eff[f_cut:]), kind='cubic')
        a_eff = 10**f(log_freq)

    # Interpolate to get system temperature.
    f = scipy.interpolate.interp1d(numpy.log10(freqs), 
        numpy.log10(t_sys), kind='cubic')
    t_sys = 10**f(log_freq)
    return a_eff, t_sys


def evaluate_scaled_noise_rms(freq_hz, num_times, bandwidth_hz=100e3, eta=1.0,
                              obs_length_h=1000.0, num_antennas=256):
    """Evaluate the station Stokes-I noise RMS, in Jy, for the given 
    observation length sampled with the specified number of times.

    Args:
        freq_hz (float):      Frequency, in Hz.
        num_times (int):      Number of time samples.
        bandwidth_hz (float): Bandwidth, in Hz.
        eta (float):          System efficiency factor.
        obs_length_h (float): Target observation length, in hours.
        num_antennas (int):   Number of antennas in a station.

    Returns:
        Noise RMS per time sample, in Jy
    """
    k_B = 1.38064852e-23
    t_acc = (obs_length_h * 3600.0) / num_times
    a_eff, t_sys = element_area_and_temperature(freq_hz)

    # Single receiver polarisation SEFD.
    sefd = (2.0 * k_B * t_sys * eta) / (a_eff * num_antennas)
    sigma_pq = (sefd * 1e26) / (2.0 * bandwidth_hz * t_acc)**0.5
    # Stokes-I noise is from two receiver polarisations so scale by 1 / sqrt(2)
    sigma_pq /= 2**0.5
    return sigma_pq


if __name__ == '__main__':
    # Global options.
    enable_noise = True
    precision = 'single'
    algorithm = 'W-projection'
    fov_deg = 2.0
    imsize = 2048
    bandwidth_hz = 100e3
    dump_time_sec = 10.0
    obs_length_h = 6.0
    lon, lat = (116.63128900, -26.69702400)  # degrees

    # Simulation dimensions.
    telescopes = [
        'models/telescope/rand_mini_fish_rmax_3.000_km.tm',
        'models/telescope/rand_mini_fish_apod_taylor-28dB_rmax_3.000_km.tm']
    freqs = [50.0, 70.0, 110.0, 170.0, 350.0, 650.0]
    pointings_az_el = [
        (  0.0, 83.0, '2014/10/01 05:43'),
        (180.0, 75.0, '2014/10/01 09:11'),
        (  0.0, 82.0, '2014/10/01 16:01'),
        (180.0, 77.0, '2014/10/01 13:06'),
        (  0.0, 79.0, '2014/10/01 20:37'),
        (  0.0, 88.0, '2014/10/01 11:42')]

    # Get telescope, frequency and pointing IDs from command line arguments.
    # These are set in the job submission script, which runs this repeatedly.
    if len(sys.argv) != 4:
        raise RuntimeError('Usage: ./run_fscn.py '
            '<telescope ID> <frequency ID> <pointing ID>')
    t = int(sys.argv[1])
    f = int(sys.argv[2])
    p = int(sys.argv[3])
    freq_hz = freqs[f] * 1e6
    output_root = 'fscn_tel' + str(t) + '_f' + str(f) + '_p' + str(p)

    # Calculate number of time steps required. Ensure it's an odd number.
    num_times = int(math.ceil(
        (obs_length_h * 3600.0) / dump_time_sec)) // 2 * 2 + 1

    # Set up the telescope model.
    print('Simulating telescope %d, frequency %.0f MHz, pointing %d' % 
        (t, freqs[f], p))
    seed = (t * len(freqs) + f) * len(pointings_az_el) + p + 1
    tel = oskar.Telescope(precision)
    tel.set_channel_bandwidth(bandwidth_hz)
    tel.set_time_average(dump_time_sec)
    tel.set_pol_mode('Scalar')
    tel.set_enable_noise(enable_noise, seed)
    tel.load(telescopes[t])

    # Set telescope thermal noise parameters.
    tel.set_noise_freq(freq_hz)
    tel.set_noise_rms(evaluate_scaled_noise_rms(freq_hz, num_times, 
        bandwidth_hz, eta=1.0, obs_length_h=obs_length_h, num_antennas=256))

    # Calculate RA, Dec and start MJD for specified pointing direction.
    az, el    = pointings_az_el[p][0], pointings_az_el[p][1]
    obs       = ephem.Observer()
    obs.lon, obs.lat, obs.elevation = math.radians(lon), math.radians(lat), 0.0
    obs.date  = pointings_az_el[p][2]
    ra, dec   = obs.radec_of(math.radians(az), math.radians(el))
    ra, dec   = math.degrees(ra), math.degrees(dec)
    mjd_mid   = ephem.julian_date(obs.date) - 2400000.5
    mjd_start = mjd_mid - obs_length_h / (2 * 24.0)
    tel.set_phase_centre(ra, dec)

    # Set up imagers for natural and uniform weighting for each sub-interval.
    intervals = numpy.logspace(numpy.log10(1), numpy.log10(num_times), 12)
    intervals = numpy.ceil(intervals).astype(int) // 2 * 2 + 1
    imagers = []
    for i in range(2 * len(intervals)):
        interval  = intervals[i // 2]
        start_idx = ((num_times - interval) // 2)
        end_idx   = ((num_times + interval) // 2) - 1
        root = output_root + ('_t%04d' % interval)
        imagers.append(oskar.Imager(precision))
        imagers[i].set(fov_deg=fov_deg, image_size=imsize, algorithm=algorithm)
        imagers[i].set(time_start=start_idx, time_end=end_idx)
        if i % 2 == 0:
            imagers[i].set(weighting='Natural', output_root=root+'_Natural')
        else:
            imagers[i].set(weighting='Uniform', output_root=root+'_Uniform')

    # Set up the imaging simulator.
    simulator = oskar.ImagingSimulator(imagers, precision)
    simulator.set_telescope_model(tel)
    simulator.set_observation_frequency(freq_hz)
    simulator.set_observation_time(start_time_mjd_utc=mjd_start,
        length_sec=obs_length_h*3600.0, num_time_steps=num_times)

    # Simulate and image.
    print('Running simulation...')
    start = time.time()
    imager_data = simulator.run(return_images=True)
    print('Completed after %.3f seconds' % (time.time() - start))

    # Write statistics to file.
    output_file = 'results_noise.txt' if enable_noise else 'results.txt'
    fh = open(output_file, 'a+')
    portalocker.lock(fh, portalocker.LOCK_EX)
    for weighting in ('Natural', 'Uniform'):
        for i in range(len(imagers)):
            if imagers[i].weighting != weighting:
                continue
            interval  = intervals[i // 2]
            rms = numpy.sqrt(numpy.mean(numpy.square(
                imager_data[i]['images'][0] )))
            fh.write('%c %d %d %d %04d %10.3e\n' % 
                (weighting[0], t, f, p, interval, rms))
    fh.close()
