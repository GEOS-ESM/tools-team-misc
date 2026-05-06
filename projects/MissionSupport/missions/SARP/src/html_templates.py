html_header = """<html>
<head>
  <link rel="stylesheet" href="https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/.internal/SARP/css/styles.css">
</head>
<body bgcolor="#000000" text="white">
<table border="0" width="100%" style="margin: 0px; margin-bottom: 40px;">
<tr>
<td align="left" valign="top" style="height: 342px; padding: 10px;" id="SARP-banner">
<p style="font-size:48px;">SARP-2026<br>Collection: $collection</p>
<p style="font-size:64px;">&emsp;</p>
<p style="font-size:36px;">$ftitle</p>
</td>
</tr>
</table>
"""

html_row_header = """<td align="left" valign="center">
<p style="font-size:24px;">&emsp;$FIELD</p>
</td>
"""

html_row = """<td align="left" valign="center">
<a class="srp-btn" href="https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/.internal/SARP/$model/$fdate/$collection/$region/movies/nasa.gmao.sarp.$collection.$field.$level.sarp.$fdate.mp4">$button</a>
</td>
"""

html_trailer = """</table>
</body>
</html>
"""

html_section = """<tr>
<th></th>
<th colspan="100" align="center" valign="center">
<p style="font-size:36px;"><br>$section<br><br></p>
</th>
</tr>
"""

sarp_stations = """<table class="table-spacing">
<tr>
<th align="center" valign="center">
<p style="font-size:36px;"><br>SARP Stations<br><br></p>
</th>
</tr>
<tr>
<th align="left" valign="bottom">
<p style="font-size:18px;">$station_plot_title</p>
</th>
</tr>
<tr>
<td align="center" valign="center">
<img src="images/stations.png" usemap="#stations" class="ImageBorder" />
</td>
</tr>
</table>

<map name="stations">
  <area shape="circle" coords="246,373,11" alt="Salton Sea" title="Salton Sea" href="$station_gram/33.23x-115.82/?pop=False&mission=SARP-2026&region=sarp-2026">
  <area shape="circle" coords="202,391,11" alt="CW3E Scripps Pier" title="CW3E Scripps Pier" href="$station_gram/32.87x-117.26/?pop=False&mission=SARP-2026&region=sarp-2026">
  <area shape="circle" coords="177,308,11" alt="Armstrong Palmdale" title="Armstrong Palmdale" href="$station_gram/34.6x-118.1/?pop=False&mission=SARP-2026&region=sarp-2026">
  <area shape="circle" coords="177,293,11" alt="Rosamond Dry Lake" title="Rosamond Dry Lake" href="$station_gram/34.9x-118.1/?pop=False&mission=SARP-2026&region=sarp-2026">
  <area shape="circle" coords="165,336,11" alt="SantaMonicaColg AERO" title="SantaMonicaColg AERO" href="$station_gram/34.0x-118.5/?pop=False&mission=SARP-2026&region=sarp-2026">
  <area shape="circle" coords="177,332,11" alt="CalTech AERO" title="CalTech AERO" href="$station_gram/34.1x-118.1/?pop=False&mission=SARP-2026&region=sarp-2026">
  <area shape="circle" coords="204,399,11" alt="San Diego" title="San Diego" href="$station_gram/32.7x-117.2/?pop=False&mission=SARP-2026&region=sarp-2026">
  <area shape="circle" coords="174,336,11" alt="Los Angeles" title="Los Angeles" href="$station_gram/34.0x-118.2/?pop=False&mission=SARP-2026&region=sarp-2026">
  <area shape="circle" coords="816,457,11" alt="Riesel Brushy Creek" title="Riesel Brushy Creek" href="$station_gram/31.48x-96.89/?pop=False&mission=SARP-2026&region=sarp-2026">
  <area shape="circle" coords="804,499,11" alt="Stiles Turkey Creek" title="Stiles Turkey Creek" href="$station_gram/30.62x-97.29/?pop=False&mission=SARP-2026&region=sarp-2026">
  <area shape="circle" coords="830,503,11" alt="TAMU Farm Brazos River" title="TAMU Farm Brazos River" href="$station_gram/30.53x-96.43/?pop=False&mission=SARP-2026&region=sarp-2026">
  <area shape="circle" coords="848,568,11" alt="Danciger Linnville Bayou" title="Danciger Linnville Bayou" href="$station_gram/29.17x-95.83/?pop=False&mission=SARP-2026&region=sarp-2026">
  <area shape="circle" coords="855,587,11" alt="Sargent Caney Creek" title="Sargent Caney Creek" href="$station_gram/28.79x-95.61/?pop=False&mission=SARP-2026&region=sarp-2026">
  <area shape="circle" coords="863,542,11" alt="Univ Houston" title="Univ Houston" href="$station_gram/29.72x-95.33/?pop=False&mission=SARP-2026&region=sarp-2026">
  <area shape="circle" coords="876,536,11" alt="Mont Belview Refineries" title="Mont Belview Refineries" href="$station_gram/29.85x-94.89/?pop=False&mission=SARP-2026&region=sarp-2026">
  <area shape="circle" coords="871,546,11" alt="Bayport Petrochem" title="Bayport Petrochem" href="$station_gram/29.63x-95.06/?pop=False&mission=SARP-2026&region=sarp-2026">
  <area shape="circle" coords="875,559,11" alt="Texas City Refineries" title="Texas City Refineries" href="$station_gram/29.37x-94.92/?pop=False&mission=SARP-2026&region=sarp-2026">
  <area shape="circle" coords="793,543,11" alt="Luling Gas Fields" title="Luling Gas Fields" href="$station_gram/29.7x-97.64/?pop=False&mission=SARP-2026&region=sarp-2026">
  <area shape="circle" coords="854,553,11" alt="WA Parish Power Plant" title="WA Parish Power Plant" href="$station_gram/29.48x-95.64/?pop=False&mission=SARP-2026&region=sarp-2026">
  <area shape="circle" coords="861,528,11" alt="Freeport Refineries" title="Freeport Refineries" href="$station_gram/30.x-95.39/?pop=False&mission=SARP-2026&region=sarp-2026">
  <area shape="circle" coords="901,526,11" alt="Beaumont Petrochem" title="Beaumont Petrochem" href="$station_gram/30.06x-94.06/?pop=False&mission=SARP-2026&region=sarp-2026">
  <area shape="circle" coords="903,528,11" alt="Beaumont Bulk Storage" title="Beaumont Bulk Storage" href="$station_gram/30.x-93.99/?pop=False&mission=SARP-2026&region=sarp-2026">
  <area shape="circle" coords="811,632,11" alt="Port Aransas Field Site" title="Port Aransas Field Site" href="$station_gram/27.84x-97.05/?pop=False&mission=SARP-2026&region=sarp-2026">
  <area shape="circle" coords="779,522,11" alt="White Family Ranch" title="White Family Ranch" href="$station_gram/30.14x-98.12/?pop=False&mission=SARP-2026&region=sarp-2026">
</map>
"""
