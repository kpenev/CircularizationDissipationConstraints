#!/usr/bin/env python3

"""Automate downloading CMD isochrone data."""

import requests
from bs4 import BeautifulSoup

if __name__ == '__main__':
    cmd_url = 'http://stev.oapd.inaf.it/cgi-bin'
    response = requests.post(
        cmd_url + '/cmd',
        {
            'cmd_version': '3.3',
            'track_parsec': 'parsec_CAF09_v1.2S',
            'track_colibri': 'parsec_CAF09_v1.2S_S35',
            'track_postagb': 'no',
            'n_inTPC': '10',
            'eta_reimers': '0.2',
            'kind_interp': '1',
            'kind_postagb': '-1',
            'photsys_file': 'tab_mag_odfnew/tab_mag_ubvrijhk.dat',
            'photsys_version': 'YBC',
            'dust_sourceM': 'nodustM',
            'dust_sourceC': 'nodustC',
            'kind_mag': '2',
            'kind_dust': '0',
            'extinction_av': '0.0',
            'extinction_coeff': 'constant',
            'extinction_curve': 'cardelli',
            'imf_file': 'tab_imf/imf_kroupa_orig.dat',
            'isoc_isagelog': '0',
            'isoc_agelow': '1.0e9',
            'isoc_ageupp': '1.0e10',
            'isoc_dage': '0.0',
            'isoc_dlage': '0.0',
            'isoc_ismetlog': '0',
            'isoc_zlow': '0.0152',
            'isoc_zupp': '0.03',
            'isoc_dz': '0.0',
            'isoc_metlow': '-2',
            'isoc_metupp': '0.3',
            'isoc_dmet': '0.0',
            'output_kind': '0',
            'output_evstage': '1',
            'lf_maginf': '-15',
            'lf_magsup': '20',
            'lf_deltamag': '0.5',
            'sim_mtot': '1.0e4',
            'output_gzip': '1',
            'submit_form': 'Submit',
            '.cgifields': ['photsys_version',
                           'isoc_ismetlog',
                           'dust_sourceC',
                           'isoc_isagelog',
                           'track_colibri',
                           'output_gzip',
                           'track_parsec',
                           'dust_sourceM',
                           'output_kind',
                           'extinction_coeff',
                           'track_postagb',
                           'extinction_curve']
        }
    )
    bs_response = BeautifulSoup(response.text, 'html.parser')
    downloaded = False
    for link in bs_response.find_all('a'):
        link_url = link.get('href')
        if link_url.endswith('.dat.gz'):
            assert not downloaded
            data_url = '%s/%s' % (cmd_url, link_url)
            print('Downloading: ' + data_url)
            with open('test_download_cmd.dat.gz', 'wb') as destination:
                destination.write(
                    requests.get(data_url, allow_redirects=True).content
                )
            print('Done')
            downloaded = True
    assert downloaded
