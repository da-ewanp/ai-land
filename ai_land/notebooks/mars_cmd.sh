#!/usr/bin/bash

mars  <<EOF
retrieve,
    expver=0001,
    class=ea,
    grid=o96,
    levtype=sfc,
    param=ssrd,
    accumulation_period=6,
    type=fc,
    date=2016-10-09,
    number=0,
    time=1800,
    step=6,
    target=out.grib
EOF

mars  <<EOF
retrieve,
    expver=0001,
    class=ea,
    grid=o96,
    levtype=sfc,
    param=ssrd,
    accumulation_period=6,
    type=fc,
    date=2016-10-08,
    number=0,
    time=1800,
    step=6,
    target=out1.grib
EOF

mars  <<EOF
retrieve,
    expver=0001,
    class=ea,
    grid=o96,
    levtype=sfc,
    param=ssrd,
    accumulation_period=6,
    type=fc,
    date=2016-10-10,
    number=0,
    time=1800,
    step=6,
    target=out2.grib
EOF