import fs from 'fs';
import path from 'path';
import glob from 'glob';
import { bbox } from "@turf/turf"
import { bindActionCreators } from "redux";
import { connect } from "react-redux";
import chroma from "chroma-js";
import Choropleth from "react-leaflet-choropleth";
import { Map, TileLayer, LayersControl, ScaleControl } from "react-leaflet";
import React, { Component } from "react";

import Control from "./Control";
import { getCsvUrl } from "../../actions/index";
import { getVectorsOnMap } from "../../actions/index";
import { getFileSuffix } from "../../actions/index";

// 
<style>
  import "font-awesome/css/font-awesome.min.css";
  import "leaflet/dist/leaflet.css";}
  import "./style.css";}
</style>

const { BaseLayer } = LayersControl;

// Color parameters for habitat risk categories and stressors
const COLOR_STEPS = 4;
const RISK_COLOR_SCALE = ['#fff', '#aa0101'];
const RISK_COLOR_HEX = chroma.scale(RISK_COLOR_SCALE).colors(COLOR_STEPS);
const RISK_VALUE_RANGE = [0, 3]; // for getting color hex for value 0 to 3
const RISK_VALUE_NAME = {
  0: 'No Score',
  1: 'Low Risk',
  2: 'Medium Risk',
  3: 'High Risk'
};
const STRESS_COLOR_SCALE = ["#d95f0e", "#d95f0e"];
const STRESS_VALUE_RANGE = [1, 1];
const RISK_FIELD_NAME = "Risk Score";
const STRESSOR_FIELD_NAME = "Stressor"

const SAMPLE_FILES = [
  // Habitat risk vectors
  "RECLASS_RISK_Ecosystem.geojson",
  "RECLASS_RISK_eelgrass.geojson",
  "RECLASS_RISK_kelp.geojson",
  "RECLASS_RISK_hardbottom.geojson",
  "RECLASS_RISK_softbottom.geojson",

  // Stressor vectors
  "STRESSOR_Docks_Wharves_Marinas.geojson",
  "STRESSOR_Finfish_Aquaculture_Comm.geojson",
  "STRESSOR_Rec_Fishing.geojson",
  "STRESSOR_Shellfish_Aquaculture_Comm.geojson",

  // Criteria score zonal statistics on each habitat-stressor pair
  "SUMMARY_STATISTICS.csv"
];

// CSV file basename.
const CSV_BASENAME = "SUMMARY_STATISTICS";

// We will only render raster files starting with these prefixes
const STRESSOR_PREFIX = "STRESSOR_"
const RISK_PREFIX = "RECLASS_RISK_"
const ECOSYSTEM_LAYER = "RECLASS_RISK_Ecosystem"

class Hramap extends Component {

  constructor(props) {
    super(props);

    this.state = {
      mapid: "mapdiv",
      coordText: "Hover mouse over the map to display coordinates.",
      maxZoom: 18,
      minZoom: 2, // global scale
      maxBbox: [[-90, -180], [90, 180]],
      lats: [],
      lngs: [],
      vectors: {},
      vectorLength: null,
      vectorsOnMap: [],
      rasters: {},
      rasterLength: null,
      rastersOnMap: [],
      // colorScale: [],
      // colorRange: [],
      ecosystemRiskLayer: ECOSYSTEM_LAYER, // Default layer on map when user choose to view sample files
      fileSuffix: "",
    };

    this.mapRef = React.createRef();
    this.layerControl = React.createRef();
    this.gatherWorkspaceFiles = this.gatherWorkspaceFiles.bind(this);
    this.loadVectors = this.loadVectors.bind(this);
  }

  componentDidMount() {
    console.log('Map Mounts');
    this.mapApi = this.mapRef.current.leafletElement; // the Leaflet Map object
    this.renderLegend();
    const fileMetadata = this.gatherWorkspaceFiles(this.props.workspace);
    this.loadVectors(fileMetadata.geojsonUrls);
    // Update csv url and file suffix reducers
    this.props.getCsvUrl(fileMetadata.csvUrl);
    this.props.getFileSuffix(fileMetadata.fileSuffix);
    // this.mapApi.invalidateSize();
    // console.log('map invalidated');
  }

  componentDidUpdate(prevProps) {
    // The map loads with the wrong center when it has been initialized
    // prior to the div that will contain it. So call this function to have 
    // the map check it's container size after that div exists.
    // console.log(prevProps);
    // if (this.props.workspace !== prevProps.workspace) {
    //   console.log('new workspace!');
      
    // }
    this.mapApi.invalidateSize();
  }

  // Read the target files from the event listener when users upload folder,
  gatherWorkspaceFiles(workspace) {
    let fileMetadata = {}
    fileMetadata.geojsonUrls = {};
    // let geotiffUrls = {};
    fileMetadata.csvUrl = null;
    fileMetadata.fileSuffix = "";

    // // Remove preexisting rasters from map
    // for (let i = 0; i < Object.keys(this.state.rastersOnMap).length; i++) {
    //   let rasterName = this.state.rastersOnMap[i];
    //   this.state.rasters[rasterName].removeFrom(this.mapApi);
    // }

    // let fileList = [];
    const files = glob.sync(path.join(workspace, 'visualization_outputs/*'));
    console.log(files);
    Object.values(files).forEach( filename => {
      // let filename = filename;
      let fileExt = filename.split(".").pop();
      let filenameNoExt = path.basename(filename).replace("." + fileExt, "");

      // Use URLs to reference to the blob, and push them to the dictionary
      if (fileExt === "geojson") {
        // Use createObjectURL() to return blob URL as a string
        fileMetadata.geojsonUrls[filenameNoExt] = path.resolve(filename) //URL.createObjectURL(filename);
      } else if (fileExt === "tif") {
        //! do not render rasters for now yet because output files are GeoJSON
        // geotiffUrls[filenameNoExt] = URL.createObjectURL(fileObj);
      } else if (
          fileExt === "csv" & filenameNoExt.startsWith(CSV_BASENAME)) {
        // Create CSV object URL
        fileMetadata.csvUrl = path.resolve(filename) //URL.createObjectURL(fileObj);
        fileMetadata.fileSuffix = filenameNoExt.slice(CSV_BASENAME.length);
      }
    });
    return(fileMetadata);

    // When all object URLs are retrieved, clean up preexisting states
    // since we will update them when new data is rendered
    // this.setState({
    //   vectors: {},
    //   vectorLength: null,
    //   vectorsOnMap: [],
    //   rasters: {},
    //   rasterLength: null,
    //   rastersOnMap: [],
    //   lats: [],
    //   lngs: [],
    //   maxBbox: [[-90, -180], [90, 180]],
    //   fileSuffix: fileSuffix,
    //   ecosystemRiskLayer: "RECLASS_RISK_Ecosystem" + fileSuffix,
    // }, () =>

    // { // Fetch vector data and create styled polygons from the URLs
    //   console.log('loading vectors');
    //   this.loadVectors(geojsonUrls);
    // });


  }


  // Read GeoJSON files and save them in the vectors state.
  loadVectors(vectorUrls) {
    let lats = [];
    let lngs = [];
    let vectorData = {};

    this.setState({vectorLength: Object.keys(vectorUrls).length});

    Object.keys(vectorUrls).forEach( vectorName => {
      // Fetch GeoJSON data via their path and store the data in vectors
      let vectorPath = vectorUrls[vectorName];
      console.log(vectorName);
      console.log(vectorPath);

      fetch(vectorPath)
        .then(response => response.json()) // parse the data as JSON
        .then(data => {

          // Get suitable color scales and field name for rendering risk and
          // stressor vectors
          let colorScale;
          let colorRange;
          if (data.name.startsWith(RISK_PREFIX)) {
            colorScale = RISK_COLOR_SCALE;
            colorRange = RISK_VALUE_RANGE;
          } else if (data.name.startsWith(STRESSOR_PREFIX)) {
            colorScale = STRESS_COLOR_SCALE;
            colorRange = STRESS_VALUE_RANGE;
          }

          // Add gradient color to features based on the field names (either
          // `Risk Score` or `stressor Potential`)
          vectorData[vectorName] = (
            <Choropleth
              key={data.name}
              data={{type: "FeatureCollection", features: data.features}}
              valueProperty={
                (feature) => {
                  // Get field values of 0 to 3 given a right vector format
                  if (vectorName.startsWith(RISK_PREFIX)) {
                    return feature.properties[RISK_FIELD_NAME];
                  } else if (vectorName.startsWith(STRESSOR_PREFIX)) {
                    return feature.properties[STRESSOR_FIELD_NAME];
                  }
                }}
              scale={colorScale}
              steps={COLOR_STEPS}
              range={colorRange}
              mode="e"  // for equidistant color mode
              style={this.getPolygonStyle()}
              onEachFeature={
                (feature, layer) => this.popupText(feature, layer)}
            />);

          // Add GeoJSON bounding box to lat and long arrays
          let vectorBbox = bbox(data);
          lngs.push(...[vectorBbox[0], vectorBbox[2]]);
          lats.push(...[vectorBbox[1], vectorBbox[3]]);

          // Only set state when all vector data are loaded
          if (Object.values(vectorData).length === this.state.vectorLength) {
            this.setState({
              // Turn on the ecosystem risk layer in the first render
              vectorsOnMap: [
                ...this.state.vectorsOnMap, this.state.ecosystemRiskLayer],
              vectors: vectorData,
              lngs: lngs,
              lats: lats
            }, () =>

            { // Update vectorsOnMap reducer
              this.props.getVectorsOnMap(this.state.vectorsOnMap);
              this.updateMaxBbox();
            });
          }
        });
    });
  }

  // Calculate the union of all the bounding boxes of geojson files.
  updateMaxBbox() {
    if (this.state.lats.length > 0 && this.state.lngs.length > 0) {
      // Calculate the min and max longitude and latitude
      let minlat = Math.min(...this.state.lats),
          maxlat = Math.max(...this.state.lats);
      let minlng = Math.min(...this.state.lngs),
          maxlng = Math.max(...this.state.lngs);
      this.setState({maxBbox: [[minlat, minlng],[maxlat, maxlng]]});
    }
  }

  // Basic style for polygons
  getPolygonStyle() {
    return {
      weight: 0.5,
      opacity: 0.8,
      color: "black",
      fillColor: "rgba(255, 255, 255, 0)",
      fillOpacity: 0.8
    };
  }

  // Pop up field names and values when user clicks on a GeoJSON feature
  popupText(feature, layer) {
    const properties = feature.properties;

    // Add text to the pop-up with all the field names & values for each feature
    Object.keys(properties).forEach( fieldName => {
      if (fieldName === RISK_FIELD_NAME) {
        let fieldValue = properties[fieldName];
        layer.bindPopup(RISK_VALUE_NAME[fieldValue]);
      } else {  // fieldName === STRESSOR_FIELD_NAME, just show the field name
        layer.bindPopup(fieldName)
      }

    });

  }

  // Render geojson vectors based on the vectorsOnMap, whose values change
  // according to the checkbox events
  renderGeojsons() {
    const vectors = this.state.vectors; // has all the vector data by name
    const vectorsOnMap = this.state.vectorsOnMap;
    let vectorsToRender = [];

    Object.keys(vectors).forEach( vectorName => {
      if (vectorsOnMap.includes(vectorName)) {
        vectorsToRender.push(vectors[vectorName]);
      }
    });

    return vectorsToRender;
  }

  // Add titles and check boxes for raster and vector layers
  addLayerControls() {
    // console.log('adding layer controls');
    // const rasters = this.state.rasters;
    const vectors = this.state.vectors;
    const vectorsOnMap = this.state.vectorsOnMap;
    const ecosystemRiskLayer = this.state.ecosystemRiskLayer;

    // Create toggle switch for Risk/stressor first
    let layerControls = [];

    const categories = {[RISK_PREFIX]: 'Habitat Risk', [STRESSOR_PREFIX]: 'Stressor'}

    if (ecosystemRiskLayer in vectors) {
      layerControls.push(
        <label key={ecosystemRiskLayer}>
          <div>
            <input
              type="radio" value={ecosystemRiskLayer} ref={ecosystemRiskLayer}
              className="leaflet-control-layers-selector"
              defaultChecked={true} // checked
              onClick={this.handleVectorLayer.bind(this)}/>

            <span className={"bold-font"}>
              {ecosystemRiskLayer.replace(RISK_PREFIX, "") + " Risk"}
            </span>
          </div>
        </label>);
    }

    for (let prefix in categories) {

      // Add the category name to the control with its first letter capitalized
      layerControls.push(
        <div key={prefix} className="control-layer-header">
          {categories[prefix]}
        </div>
      );

      // If vector name starts with 'RECLASS_RISK' or 'STRESSOR_', put them on
      // the control box and remove the front suffix from the names
      Object.keys(vectors).forEach( vectorName => {
        // console.log(vectorName);
        // If layer name is in the vectorsOnMap array, set it checked
        if (vectorName.startsWith(prefix) & vectorName !== ecosystemRiskLayer) {
          let defaultCheck = false; // Boolean variable for setting default layer check box
          if (vectorsOnMap.includes(vectorName)) {
            let defaultCheck = true;
          }
          layerControls.push(
            <label key={vectorName}>
              <div>
                <input
                  type="checkbox" value={vectorName} ref={vectorName}
                  className="leaflet-control-layers-selector"
                  defaultChecked={defaultCheck}
                  onClick={this.handleVectorLayer.bind(this)}/>
                <span>
                  {vectorName.replace(prefix, "")}
                </span>
              </div>
            </label>);
        }
      });
    }

    // When the control has layers, not just the category names
    if (layerControls.length > 2) {
      return (
        <Control position="topright">
          <div
           className="leaflet-control-layers
                      leaflet-control-layers-expanded indent-box">
            {layerControls}
          </div>
        </Control>
      );
    }
  }

  // Turn raster layers on or off depending on the checkbox value
  handleRasterLayer(e) {
    const checkBoxValue = e.target.value;
    if (this.state.rastersOnMap.includes(checkBoxValue)) {
      this.setState({rastersOnMap: this.state.rastersOnMap.filter(
        raster => raster !== checkBoxValue)});
      this.state.rasters[checkBoxValue].removeFrom(this.mapApi);
    } else {
      this.setState({rastersOnMap: [
        ...this.state.rastersOnMap, checkBoxValue]});
      this.state.rasters[checkBoxValue].addTo(this.mapApi);
    }
  }

  // Turn vector layers on or off depending on the checkbox value
  handleVectorLayer(e) {
    const checkBoxValue = e.target.value;
    const vectorsOnMap = this.state.vectorsOnMap;
    let updatedVectorsOnMap = [...this.state.vectorsOnMap];
    const ecosystemRiskLayer = this.state.ecosystemRiskLayer;

    // Turn off the layer if it's on map
    if (vectorsOnMap.includes(checkBoxValue)) {
      let index = updatedVectorsOnMap.indexOf(checkBoxValue);

      if (checkBoxValue === ecosystemRiskLayer) {
        this.refs[ecosystemRiskLayer].checked = false; // Uncheck the ecosystem layer
      }

      // Remove the layer from the list
      if (index !== -1) {
        updatedVectorsOnMap.splice(index, 1);
      }

    } else {
      // Turn off all the other risk layers if ecosystem risk layer is checked
      if (checkBoxValue === ecosystemRiskLayer) {
         for (let i = updatedVectorsOnMap.length - 1; i >= 0; i--) {
          let layerName = updatedVectorsOnMap[i];

          if (layerName.startsWith(RISK_PREFIX)) {
            let index = updatedVectorsOnMap.indexOf(layerName);
            updatedVectorsOnMap.splice(index, 1);
            this.refs[layerName].checked = false; // Uncheck the layer from control
          }
         }
      } else if (vectorsOnMap.includes(ecosystemRiskLayer) &
                 checkBoxValue.startsWith(RISK_PREFIX)) {
        // Turn off the ecosystem layer if a habitat risk layer is checked
        let index = updatedVectorsOnMap.indexOf(ecosystemRiskLayer);
        updatedVectorsOnMap.splice(index, 1);
        this.refs[ecosystemRiskLayer].checked = false;
      }
      // Turn on the layer if it's not on map
      updatedVectorsOnMap.push(checkBoxValue);
    }

    this.setState({vectorsOnMap: updatedVectorsOnMap}, () => {
      // Update vectorsOnMap reducer
      this.props.getVectorsOnMap(this.state.vectorsOnMap);
    });

  }

  // Display lat and long coordinates based on mouse event
  renderMouseCoords(e) {
    let coordText = "Lat: " + e.latlng.lat.toFixed(5) +
                 ", Long: "+e.latlng.lng.toFixed(5);
    this.setState({ coordText });
  }

  // Remove the display of coords when mouse leaves the map
  removeCoords(e) {
    this.setState({coords: "Hover mouse over the map to display coordinates."});
  }

  // Zoom to the union of bounding boxes of all layers
  zoomToMaxBbox() {
    this.mapApi.fitBounds(this.state.maxBbox);
  }

  // Display the risk and resilience legends
  renderLegend() {

    let legend = [];

    // Risk legend
    Object.keys(RISK_VALUE_NAME).forEach( riskValue => {
      let riskName = RISK_VALUE_NAME[riskValue];
      let riskColorHex = RISK_COLOR_HEX[riskValue];
      // Set rect size here, because the rect css does not work in firefox
      legend.push(
        <li key={riskName}>
            <svg className='legend-svg'>
              <rect className='legend-svg' width='28px' height='18px'
                    fill={riskColorHex}/>
            </svg>
            <span className='legend-text'>{riskName}</span>
        </li>
      );
    });

    // stressor legend
    legend.push(
      <li key="stressor">
          <svg className='legend-svg'>
            <rect className='legend-svg' width='28px' height='18px'
                  fill={STRESS_COLOR_SCALE[0]}/>
          </svg>
          <span className='legend-text'>Stressor</span>
      </li>
    );

    this.setState({ legend });
  }



  render() {
    return (
        <Map ref={this.mapRef} id={this.state.mapid} className="map"
          maxZoom={this.state.maxZoom} minZoom={this.state.minZoom}
          onMouseMove={this.renderMouseCoords.bind(this)}
          onMouseOut={this.removeCoords.bind(this)}
          bounds={this.state.maxBbox}>

        <Control position="topleft">
          <button id="zoom-btn" title="Zoom To AOI" aria-label="Zoom To AOI"
            onClick={this.zoomToMaxBbox.bind(this)}>
            <i className="fa fa-crosshairs fa-lg"/>
          </button>
        </Control>

        <LayersControl position="topleft">
          <BaseLayer name="Open Street Map" checked={true}>
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution="&copy; <a href=&quot;http://osm.org/copyright&quot;>
                OpenStreetMap</a> contributors" />
          </BaseLayer>

          <BaseLayer name="ESRI Ocean Map">
            <TileLayer
              url="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean_Basemap/MapServer/tile/{z}/{y}/{x}"
              attribution="&copy;
                Esri, GEBCO, NOAA, National Geographic, DeLorme, HERE, Geonames.org, and other contributors"/>
          </BaseLayer>

          <BaseLayer name="ESRI World Imagery">
            <TileLayer
              url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
              attribution="&copy;
                Esri, DigitalGlobe, Earthstar Geographics, CNES/Airbus DS,
                GeoEye, USDA FSA, USGS, Aerogrid, IGN, IGP, and the GIS User Community"/>
          </BaseLayer>
        </LayersControl>

        {this.renderGeojsons()}

        {this.addLayerControls()}

{/*        <Control position="topleft">
          // <form className="leaflet-control-layers upload-form">

          //   Upload HRA Output Folder
          //   <input type="file" ref="fileUpload"
          //     onChange={e => this.onWorkspaceReady(e.target.files)}
          //     webkitdirectory="true" mozdirectory="true" msdirectory="true"
          //     odirectory="true" directory="true"
          //     style={{"display": "none"}} multiple />
          //   <br />

          //   <input type="button" value="Select Folder"
          //     onClick={e => this.refs.fileUpload.click()}/>

          //   or View Sample
          //   <input type="button" value="Sample Files"
          //     onClick={e => this.viewSample(e)}/>
          // </form>

        </Control>
*/}
        <Control position="bottomleft">
          <div className="coords">{this.state.coordText}</div>
        </Control>

        <Control position='bottomright' key='legend'>
          <ul className='legend'>
            {this.state.legend}
          </ul>
        </Control>

        <ScaleControl position={"bottomleft"} maxWidth={100}/>
        </Map>
    );
  }
}

// Use redux API to create csvUrl & vectorsOnMap props connected to the redux state
function mapStateToProps(state) {
  return {
    csvUrl: state.csvUrl,
    vectorsOnMap: state.vectorsOnMap,
    fileSuffix: state.fileSuffix,
  }
}

// Return getCsvUrl, getVectorsOnMap & getFileSuffix callback props
function mapDispatchToProps(dispatch) {
  return bindActionCreators({ getCsvUrl, getVectorsOnMap, getFileSuffix }, dispatch);
}

export default connect(mapStateToProps, mapDispatchToProps)(Hramap);
