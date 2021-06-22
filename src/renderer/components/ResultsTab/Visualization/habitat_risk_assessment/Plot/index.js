import React, { Component } from 'react';
import { connect } from 'react-redux';
import Papa from 'papaparse';
import palette from 'google-palette';
import { ScatterplotChart } from 'react-easy-chart';
import Legend from './Legend';
import ToolTip from './ToolTip';

// Header names in stats CSV file
const HABITAT_HEADER = 'HABITAT';
const STRESSOR_HEADER = 'STRESSOR';
const EXPOSURE_HEADER = 'E_MEAN';
const CONSEQUENCE_HEADER = 'C_MEAN';
const SUBREGION_HEADER = 'SUBREGION'
const HIGH_RISK_HEADER = 'R_%HIGH';
const MED_RISK_HEADER = 'R_%MEDIUM';
const LOW_RISK_HEADER = 'R_%LOW';

// Default layer on map when files are uploaded
const ECOSYSTEM_RISK_LAYER = "RECLASS_RISK_Ecosystem";

// Prefix of risk rasters
const RISK_PREFIX = "RECLASS_RISK_"

class riskPlots extends Component {

  constructor(props) {
    super(props);
    this.state = {
      csvUrl: props.csvUrl, // provided by connect@mapStateToProps
      vectorsOnMap: props.vectorsOnMap, // same as above
      fileSuffix: props.fileSuffix,
      showToolTip: false,
      dataToRender: {},
      colorConfig: [],
      legendColorConfig: [],
    };
  }

  componentWillReceiveProps(nextProps) {
    if (nextProps.csvUrl !== null & nextProps.csvUrl !== undefined) {
      this.setState({
         csvUrl: nextProps.csvUrl,
         vectorsOnMap: nextProps.vectorsOnMap,
         fileSuffix: nextProps.fileSuffix,
      }, () => {
        // Parse zonal statistics data from CSV
        Papa.parse(nextProps.csvUrl, {
          header: true,
          download: true,
          skipEmptyLines: true,
          complete: results => this.convertData(results.data)
        });
      });
    }
  }

  // Process CSV data to dataToRender state for ScatterplotChart to render
  convertData(csvData) {
    // Use vectorsOnMap as a reference for what data to render
    const vectorsOnMap = this.state.vectorsOnMap;
    const fileSuffix = this.state.fileSuffix;

    // Get a dictionary of subregions as keys and data points as values
    let dataToRender = {};

    // Get habitats on map in order to generate color palette in plots
    let habitatsOnMap = [];

    // Get a list of all the habitats
    let habitatsAll = [];

    Object.values(csvData).forEach( row => {
      let subregion = row[SUBREGION_HEADER];
      // Add suffix to habiat header, because all vector files will have
      // suffix appended.
      let habitatName = row[HABITAT_HEADER] + fileSuffix;

      // Add habitat name to the all habitats list
      if (!habitatsAll.includes(habitatName)) {
        habitatsAll.push(habitatName); // add habitat name to the array
      }

      // Render data points for that habitat layer if it's checked or if
      // ECOSYSTEM_RISK_LAYER is checked on map control
      if (vectorsOnMap.includes(RISK_PREFIX + habitatName) ||
          vectorsOnMap.includes(ECOSYSTEM_RISK_LAYER + fileSuffix)) {

        // Add habitat name to the legend list
        if (!habitatsOnMap.includes(habitatName)) {
          habitatsOnMap.push(habitatName); // add habitat name to the array
        }

        // Add the data point for the stressor and habitat to state
        let dataPoint = {
          type: habitatName,
          stressor: row[STRESSOR_HEADER],
          x: row[EXPOSURE_HEADER],
          y: row[CONSEQUENCE_HEADER],
          highRisk: row[HIGH_RISK_HEADER],
          medRisk: row[MED_RISK_HEADER],
          lowRisk: row[LOW_RISK_HEADER]
        };

        // Push data to the array for the subregion key in dataToRender
        if (!dataToRender[subregion]) {
          dataToRender[subregion] = [dataPoint];
        }
        else {
          dataToRender[subregion].push(dataPoint);
        }
      }
    });

    this.setState({ dataToRender, habitatsAll, habitatsOnMap });
    this.createColorConfig();
  }

  // Create color hexes for habitats in scatter plots and legend
  createColorConfig() {
    const habitatsAll = this.state.habitatsAll;
    const habitatsOnMap = this.state.habitatsOnMap;

    // Get Paul Tol's rainbow scheme, color-blind friendly
    const colors = palette('tol', habitatsAll.length)

    let colorConfig = [];
    let legendColorConfig = [];

    for (let i=0; i < habitatsAll.length; i++) {
      let habitatName = habitatsAll[i]
      // Push every habitat to the color config for plots, so the plots
      // have consistent color scheme for habitats
      colorConfig.push(
        {
          type: habitatName,
          color: '#' + colors[i]
        }
      );

      // Only push habitats that are on the map to the legend
      if (habitatsOnMap.includes(habitatName)) {
        legendColorConfig.push(
          {
            type: habitatName,
            color: '#' + colors[i]
          }
        );
      }
    }

    if (colorConfig.length === habitatsAll.length) {
      this.setState({ colorConfig });
      this.setState({ legendColorConfig });
    }
  }

  createScatterPlot() {
    if ( this.state.dataToRender ) {
      const dataToRender = this.state.dataToRender;
      const subregions = Object.keys(dataToRender);
      const scatterPlots = subregions.map( subregion => {
        // Find max x and y values among all the data points
        const x_max = Math.max.apply(
          Math, Object.values(
            dataToRender[subregion]).map(dataPoint => dataPoint.x));
        const y_max = Math.max.apply(
          Math, Object.values(
            dataToRender[subregion]).map(dataPoint => dataPoint.y));
        // Make extra space for the plot by multiplying 1.1
        const max_domain = Math.max(x_max, y_max) * 1.1;

        return (<div key={subregion}>
          <h4>{subregion}</h4>
          <ScatterplotChart
            key={subregion}
            data={dataToRender[subregion]}
            margin={{top: 10, right: 0, bottom: 25, left: 65}}
            config={this.state.colorConfig}
            axes
            xTicks={7}
            yTicks={7}
            axisLabels={{x: 'Exposure', y: 'Consequence'}}
            dotRadius={6}
            width={400}
            height={300}
            xDomainRange={[0, max_domain]}
            yDomainRange={[0, max_domain]}
            mouseOverHandler={this.mouseOverHandler.bind(this)}
            mouseOutHandler={this.mouseOutHandler.bind(this)}
          />
        </div>)
      });

      return scatterPlots
    }
  }

  // Show tool tip on top of data point when mouse hovers over
  mouseOverHandler(d, e) {
    this.setState({
      showToolTip: true,
      top: `${e.clientY - 60}px`,
      left: `${e.clientX - d.stressor.length * 3.5}px`,  // roughly center the element
      habitatName: d.type,
      stressorName: d.stressor,
      conseqValue: Number(d.y),
      expoValue: Number(d.x)
    });
  }

  mouseOutHandler() {
    this.setState({ showToolTip: false });
  }

  createTooltip() {
    if (this.state.showToolTip) {
      return (
        <ToolTip top={this.state.top} left={this.state.left}>
          <span className='tooltip-title'> {this.state.stressorName} </span>
          <br />
          <span className='tooltip-title'> {this.state.habitatName} </span>
          <br />
          E: {this.state.expoValue.toFixed(2)},
          C: {this.state.conseqValue.toFixed(2)}

        </ToolTip>
      );
    }
    return false;
  }

  createLegend() {
    if ( Object.values(this.state.dataToRender).length > 0 ) {
      return(
        <div className='legend-div'>
          <Legend config={this.state.legendColorConfig} />
        </div>
      );
    }
  }

  defaultParagraph() {
    const welcomeText = "Welcome to the Habitat Risk Assessment " +
      "visualization platform!"

    const explainText = "Please click on the button on the left of the map " +
       "to upload your output data folder from the HRA model, in order to " +
       "view the results on map and plots. You can also view sample files " +
       "to expect what they look like.";

    if (!this.state.csvUrl) {
      return (
        <div className='default-p'>
          <p>{welcomeText}</p>
          <p>{explainText}</p>
        </div>
      );
    } else {
      return null
    }
  }

  render() {

    return (
      <div className='plot-container'>
        <h3 className='header'>Risk Plots</h3>

        {this.createLegend()}

        {this.defaultParagraph()}

        {this.createScatterPlot()}

        {this.createTooltip()}

      </div>
    );
  }
}

const mapStateToProps = state => ({
  csvUrl: state.csvUrl,
  vectorsOnMap: state.vectorsOnMap,
  fileSuffix: state.fileSuffix,
});

export default connect(mapStateToProps)(riskPlots);
