// This is from https://github.com/LiveBy/react-leaflet-control/issues/27.
// the codesandbox snippet by bslipek to solve the issue that the contents
// of <Control> do not get rendered on initial <Map> display.

import ReactDOM from "react-dom";
import { MapControl, withLeaflet } from "react-leaflet";
import { Control, DomUtil, DomEvent } from "leaflet";

const DumbControl = Control.extend({
  options: {
    className: "",
    onOff: "",
    handleOff: function noop() {}
  },

  onAdd(/* map */) {
    var _controlDiv = DomUtil.create("div", this.options.className);
    DomEvent.disableClickPropagation(_controlDiv);
    return _controlDiv;
  },

  onRemove(map) {
    if (this.options.onOff) {
      map.off(this.options.onOff, this.options.handleOff, this);
    }

    return this;
  }
});

export default withLeaflet(
  class LeafletControl extends MapControl {
    createLeafletElement(props) {
      return new DumbControl(Object.assign({}, props));
    }

    componentDidMount() {
      // Here is a little trick
      this.forceUpdate();
      this.leafletElement.addTo(this.props.leaflet.map);
      // for leaflet 1.x
      // this.leafletElement.addTo(this.context.map)
    }

    render() {
      if (!this.leafletElement || !this.leafletElement.getContainer()) {
        return null;
      }
      return ReactDOM.createPortal(
        this.props.children,
        this.leafletElement.getContainer()
      );
    }
  }
);
