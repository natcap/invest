<!DOCTYPE html>
<html lang="en">
<head>
  <meta http-equiv="X-UA-Compatible" content="IE=edge" />
  <meta charset="utf-8"><script type="text/javascript">(window.NREUM||(NREUM={})).loader_config={xpid:"VwMGVVZSGwIIUFBQDwU="};window.NREUM||(NREUM={}),__nr_require=function(t,e,n){function r(n){if(!e[n]){var o=e[n]={exports:{}};t[n][0].call(o.exports,function(e){var o=t[n][1][e];return r(o?o:e)},o,o.exports)}return e[n].exports}if("function"==typeof __nr_require)return __nr_require;for(var o=0;o<n.length;o++)r(n[o]);return r}({QJf3ax:[function(t,e){function n(t){function e(e,n,a){t&&t(e,n,a),a||(a={});for(var c=s(e),f=c.length,u=i(a,o,r),d=0;f>d;d++)c[d].apply(u,n);return u}function a(t,e){f[t]=s(t).concat(e)}function s(t){return f[t]||[]}function c(){return n(e)}var f={};return{on:a,emit:e,create:c,listeners:s,_events:f}}function r(){return{}}var o="nr@context",i=t("gos");e.exports=n()},{gos:"7eSDFh"}],ee:[function(t,e){e.exports=t("QJf3ax")},{}],3:[function(t){function e(t){try{i.console&&console.log(t)}catch(e){}}var n,r=t("ee"),o=t(1),i={};try{n=localStorage.getItem("__nr_flags").split(","),console&&"function"==typeof console.log&&(i.console=!0,-1!==n.indexOf("dev")&&(i.dev=!0),-1!==n.indexOf("nr_dev")&&(i.nrDev=!0))}catch(a){}i.nrDev&&r.on("internal-error",function(t){e(t.stack)}),i.dev&&r.on("fn-err",function(t,n,r){e(r.stack)}),i.dev&&(e("NR AGENT IN DEVELOPMENT MODE"),e("flags: "+o(i,function(t){return t}).join(", ")))},{1:23,ee:"QJf3ax"}],4:[function(t){function e(t,e,n,i,s){try{c?c-=1:r("err",[s||new UncaughtException(t,e,n)])}catch(f){try{r("ierr",[f,(new Date).getTime(),!0])}catch(u){}}return"function"==typeof a?a.apply(this,o(arguments)):!1}function UncaughtException(t,e,n){this.message=t||"Uncaught error with no additional information",this.sourceURL=e,this.line=n}function n(t){r("err",[t,(new Date).getTime()])}var r=t("handle"),o=t(6),i=t("ee"),a=window.onerror,s=!1,c=0;t("loader").features.err=!0,t(5),window.onerror=e;try{throw new Error}catch(f){"stack"in f&&(t(1),t(2),"addEventListener"in window&&t(3),window.XMLHttpRequest&&XMLHttpRequest.prototype&&XMLHttpRequest.prototype.addEventListener&&window.XMLHttpRequest&&XMLHttpRequest.prototype&&XMLHttpRequest.prototype.addEventListener&&!/CriOS/.test(navigator.userAgent)&&t(4),s=!0)}i.on("fn-start",function(){s&&(c+=1)}),i.on("fn-err",function(t,e,r){s&&(this.thrown=!0,n(r))}),i.on("fn-end",function(){s&&!this.thrown&&c>0&&(c-=1)}),i.on("internal-error",function(t){r("ierr",[t,(new Date).getTime(),!0])})},{1:10,2:9,3:7,4:11,5:3,6:24,ee:"QJf3ax",handle:"D5DuLP",loader:"G9z0Bl"}],5:[function(t){t("loader").features.ins=!0},{loader:"G9z0Bl"}],6:[function(t){function e(){}if(window.performance&&window.performance.timing&&window.performance.getEntriesByType){var n=t("ee"),r=t("handle"),o=t(1),i=t(2);t("loader").features.stn=!0,t(3),n.on("fn-start",function(t){var e=t[0];e instanceof Event&&(this.bstStart=Date.now())}),n.on("fn-end",function(t,e){var n=t[0];n instanceof Event&&r("bst",[n,e,this.bstStart,Date.now()])}),o.on("fn-start",function(t,e,n){this.bstStart=Date.now(),this.bstType=n}),o.on("fn-end",function(t,e){r("bstTimer",[e,this.bstStart,Date.now(),this.bstType])}),i.on("fn-start",function(){this.bstStart=Date.now()}),i.on("fn-end",function(t,e){r("bstTimer",[e,this.bstStart,Date.now(),"requestAnimationFrame"])}),n.on("pushState-start",function(){this.time=Date.now(),this.startPath=location.pathname+location.hash}),n.on("pushState-end",function(){r("bstHist",[location.pathname+location.hash,this.startPath,this.time])}),"addEventListener"in window.performance&&(window.performance.addEventListener("webkitresourcetimingbufferfull",function(){r("bstResource",[window.performance.getEntriesByType("resource")]),window.performance.webkitClearResourceTimings()},!1),window.performance.addEventListener("resourcetimingbufferfull",function(){r("bstResource",[window.performance.getEntriesByType("resource")]),window.performance.clearResourceTimings()},!1)),document.addEventListener("scroll",e,!1),document.addEventListener("keypress",e,!1),document.addEventListener("click",e,!1)}},{1:10,2:9,3:8,ee:"QJf3ax",handle:"D5DuLP",loader:"G9z0Bl"}],7:[function(t,e){function n(t){i.inPlace(t,["addEventListener","removeEventListener"],"-",r)}function r(t){return t[1]}var o=(t(1),t("ee").create()),i=t(2)(o),a=t("gos");if(e.exports=o,n(window),"getPrototypeOf"in Object){for(var s=document;s&&!s.hasOwnProperty("addEventListener");)s=Object.getPrototypeOf(s);s&&n(s);for(var c=XMLHttpRequest.prototype;c&&!c.hasOwnProperty("addEventListener");)c=Object.getPrototypeOf(c);c&&n(c)}else XMLHttpRequest.prototype.hasOwnProperty("addEventListener")&&n(XMLHttpRequest.prototype);o.on("addEventListener-start",function(t){if(t[1]){var e=t[1];"function"==typeof e?this.wrapped=t[1]=a(e,"nr@wrapped",function(){return i(e,"fn-",null,e.name||"anonymous")}):"function"==typeof e.handleEvent&&i.inPlace(e,["handleEvent"],"fn-")}}),o.on("removeEventListener-start",function(t){var e=this.wrapped;e&&(t[1]=e)})},{1:24,2:25,ee:"QJf3ax",gos:"7eSDFh"}],8:[function(t,e){var n=(t(2),t("ee").create()),r=t(1)(n);e.exports=n,r.inPlace(window.history,["pushState"],"-")},{1:25,2:24,ee:"QJf3ax"}],9:[function(t,e){var n=(t(2),t("ee").create()),r=t(1)(n);e.exports=n,r.inPlace(window,["requestAnimationFrame","mozRequestAnimationFrame","webkitRequestAnimationFrame","msRequestAnimationFrame"],"raf-"),n.on("raf-start",function(t){t[0]=r(t[0],"fn-")})},{1:25,2:24,ee:"QJf3ax"}],10:[function(t,e){function n(t,e,n){t[0]=o(t[0],"fn-",null,n)}var r=(t(2),t("ee").create()),o=t(1)(r);e.exports=r,o.inPlace(window,["setTimeout","setInterval","setImmediate"],"setTimer-"),r.on("setTimer-start",n)},{1:25,2:24,ee:"QJf3ax"}],11:[function(t,e){function n(){f.inPlace(this,p,"fn-")}function r(t,e){f.inPlace(e,["onreadystatechange"],"fn-")}function o(t,e){return e}function i(t,e){for(var n in t)e[n]=t[n];return e}var a=t("ee").create(),s=t(1),c=t(2),f=c(a),u=c(s),d=window.XMLHttpRequest,p=["onload","onerror","onabort","onloadstart","onloadend","onprogress","ontimeout"];e.exports=a,window.XMLHttpRequest=function(t){var e=new d(t);try{a.emit("new-xhr",[],e),u.inPlace(e,["addEventListener","removeEventListener"],"-",o),e.addEventListener("readystatechange",n,!1)}catch(r){try{a.emit("internal-error",[r])}catch(i){}}return e},i(d,XMLHttpRequest),XMLHttpRequest.prototype=d.prototype,f.inPlace(XMLHttpRequest.prototype,["open","send"],"-xhr-",o),a.on("send-xhr-start",r),a.on("open-xhr-start",r)},{1:7,2:25,ee:"QJf3ax"}],12:[function(t){function e(t){var e=this.params,r=this.metrics;if(!this.ended){this.ended=!0;for(var i=0;c>i;i++)t.removeEventListener(s[i],this.listener,!1);if(!e.aborted){if(r.duration=(new Date).getTime()-this.startTime,4===t.readyState){e.status=t.status;var a=t.responseType,f="arraybuffer"===a||"blob"===a||"json"===a?t.response:t.responseText,u=n(f);if(u&&(r.rxSize=u),this.sameOrigin){var d=t.getResponseHeader("X-NewRelic-App-Data");d&&(e.cat=d.split(", ").pop())}}else e.status=0;r.cbTime=this.cbTime,o("xhr",[e,r,this.startTime])}}}function n(t){if("string"==typeof t&&t.length)return t.length;if("object"!=typeof t)return void 0;if("undefined"!=typeof ArrayBuffer&&t instanceof ArrayBuffer&&t.byteLength)return t.byteLength;if("undefined"!=typeof Blob&&t instanceof Blob&&t.size)return t.size;if("undefined"!=typeof FormData&&t instanceof FormData)return void 0;try{return JSON.stringify(t).length}catch(e){return void 0}}function r(t,e){var n=i(e),r=t.params;r.host=n.hostname+":"+n.port,r.pathname=n.pathname,t.sameOrigin=n.sameOrigin}if(window.XMLHttpRequest&&XMLHttpRequest.prototype&&XMLHttpRequest.prototype.addEventListener&&!/CriOS/.test(navigator.userAgent)){t("loader").features.xhr=!0;var o=t("handle"),i=t(2),a=t("ee"),s=["load","error","abort","timeout"],c=s.length,f=t(1);t(4),t(3),a.on("new-xhr",function(){this.totalCbs=0,this.called=0,this.cbTime=0,this.end=e,this.ended=!1,this.xhrGuids={}}),a.on("open-xhr-start",function(t){this.params={method:t[0]},r(this,t[1]),this.metrics={}}),a.on("open-xhr-end",function(t,e){"loader_config"in NREUM&&"xpid"in NREUM.loader_config&&this.sameOrigin&&e.setRequestHeader("X-NewRelic-ID",NREUM.loader_config.xpid)}),a.on("send-xhr-start",function(t,e){var r=this.metrics,o=t[0],i=this;if(r&&o){var f=n(o);f&&(r.txSize=f)}this.startTime=(new Date).getTime(),this.listener=function(t){try{"abort"===t.type&&(i.params.aborted=!0),("load"!==t.type||i.called===i.totalCbs&&(i.onloadCalled||"function"!=typeof e.onload))&&i.end(e)}catch(n){try{a.emit("internal-error",[n])}catch(r){}}};for(var u=0;c>u;u++)e.addEventListener(s[u],this.listener,!1)}),a.on("xhr-cb-time",function(t,e,n){this.cbTime+=t,e?this.onloadCalled=!0:this.called+=1,this.called!==this.totalCbs||!this.onloadCalled&&"function"==typeof n.onload||this.end(n)}),a.on("xhr-load-added",function(t,e){var n=""+f(t)+!!e;this.xhrGuids&&!this.xhrGuids[n]&&(this.xhrGuids[n]=!0,this.totalCbs+=1)}),a.on("xhr-load-removed",function(t,e){var n=""+f(t)+!!e;this.xhrGuids&&this.xhrGuids[n]&&(delete this.xhrGuids[n],this.totalCbs-=1)}),a.on("addEventListener-end",function(t,e){e instanceof XMLHttpRequest&&"load"===t[0]&&a.emit("xhr-load-added",[t[1],t[2]],e)}),a.on("removeEventListener-end",function(t,e){e instanceof XMLHttpRequest&&"load"===t[0]&&a.emit("xhr-load-removed",[t[1],t[2]],e)}),a.on("fn-start",function(t,e,n){e instanceof XMLHttpRequest&&("onload"===n&&(this.onload=!0),("load"===(t[0]&&t[0].type)||this.onload)&&(this.xhrCbStart=(new Date).getTime()))}),a.on("fn-end",function(t,e){this.xhrCbStart&&a.emit("xhr-cb-time",[(new Date).getTime()-this.xhrCbStart,this.onload,e],e)})}},{1:"XL7HBI",2:13,3:11,4:7,ee:"QJf3ax",handle:"D5DuLP",loader:"G9z0Bl"}],13:[function(t,e){e.exports=function(t){var e=document.createElement("a"),n=window.location,r={};e.href=t,r.port=e.port;var o=e.href.split("://");return!r.port&&o[1]&&(r.port=o[1].split("/")[0].split("@").pop().split(":")[1]),r.port&&"0"!==r.port||(r.port="https"===o[0]?"443":"80"),r.hostname=e.hostname||n.hostname,r.pathname=e.pathname,r.protocol=o[0],"/"!==r.pathname.charAt(0)&&(r.pathname="/"+r.pathname),r.sameOrigin=!e.hostname||e.hostname===document.domain&&e.port===n.port&&e.protocol===n.protocol,r}},{}],14:[function(t,e){function n(t){return function(){r(t,[(new Date).getTime()].concat(i(arguments)))}}var r=t("handle"),o=t(1),i=t(2);"undefined"==typeof window.newrelic&&(newrelic=window.NREUM);var a=["setPageViewName","addPageAction","setCustomAttribute","finished","addToTrace","inlineHit","noticeError"];o(a,function(t,e){window.NREUM[e]=n("api-"+e)}),e.exports=window.NREUM},{1:23,2:24,handle:"D5DuLP"}],"7eSDFh":[function(t,e){function n(t,e,n){if(r.call(t,e))return t[e];var o=n();if(Object.defineProperty&&Object.keys)try{return Object.defineProperty(t,e,{value:o,writable:!0,enumerable:!1}),o}catch(i){}return t[e]=o,o}var r=Object.prototype.hasOwnProperty;e.exports=n},{}],gos:[function(t,e){e.exports=t("7eSDFh")},{}],handle:[function(t,e){e.exports=t("D5DuLP")},{}],D5DuLP:[function(t,e){function n(t,e,n){return r.listeners(t).length?r.emit(t,e,n):(o[t]||(o[t]=[]),void o[t].push(e))}var r=t("ee").create(),o={};e.exports=n,n.ee=r,r.q=o},{ee:"QJf3ax"}],id:[function(t,e){e.exports=t("XL7HBI")},{}],XL7HBI:[function(t,e){function n(t){var e=typeof t;return!t||"object"!==e&&"function"!==e?-1:t===window?0:i(t,o,function(){return r++})}var r=1,o="nr@id",i=t("gos");e.exports=n},{gos:"7eSDFh"}],G9z0Bl:[function(t,e){function n(){var t=p.info=NREUM.info,e=f.getElementsByTagName("script")[0];if(t&&t.licenseKey&&t.applicationID&&e){s(d,function(e,n){e in t||(t[e]=n)});var n="https"===u.split(":")[0]||t.sslForHttp;p.proto=n?"https://":"http://",a("mark",["onload",i()]);var r=f.createElement("script");r.src=p.proto+t.agent,e.parentNode.insertBefore(r,e)}}function r(){"complete"===f.readyState&&o()}function o(){a("mark",["domContent",i()])}function i(){return(new Date).getTime()}var a=t("handle"),s=t(1),c=(t(2),window),f=c.document,u=(""+location).split("?")[0],d={beacon:"bam.nr-data.net",errorBeacon:"bam.nr-data.net",agent:"js-agent.newrelic.com/nr-632.min.js"},p=e.exports={offset:i(),origin:u,features:{}};f.addEventListener?(f.addEventListener("DOMContentLoaded",o,!1),c.addEventListener("load",n,!1)):(f.attachEvent("onreadystatechange",r),c.attachEvent("onload",n)),a("mark",["firstbyte",i()])},{1:23,2:14,handle:"D5DuLP"}],loader:[function(t,e){e.exports=t("G9z0Bl")},{}],23:[function(t,e){function n(t,e){var n=[],o="",i=0;for(o in t)r.call(t,o)&&(n[i]=e(o,t[o]),i+=1);return n}var r=Object.prototype.hasOwnProperty;e.exports=n},{}],24:[function(t,e){function n(t,e,n){e||(e=0),"undefined"==typeof n&&(n=t?t.length:0);for(var r=-1,o=n-e||0,i=Array(0>o?0:o);++r<o;)i[r]=t[e+r];return i}e.exports=n},{}],25:[function(t,e){function n(t){return!(t&&"function"==typeof t&&t.apply&&!t[i])}var r=t("ee"),o=t(1),i="nr@wrapper",a=Object.prototype.hasOwnProperty;e.exports=function(t){function e(t,e,r,a){function nrWrapper(){var n,i,s,f;try{i=this,n=o(arguments),s=r&&r(n,i)||{}}catch(d){u([d,"",[n,i,a],s])}c(e+"start",[n,i,a],s);try{return f=t.apply(i,n)}catch(p){throw c(e+"err",[n,i,p],s),p}finally{c(e+"end",[n,i,f],s)}}return n(t)?t:(e||(e=""),nrWrapper[i]=!0,f(t,nrWrapper),nrWrapper)}function s(t,r,o,i){o||(o="");var a,s,c,f="-"===o.charAt(0);for(c=0;c<r.length;c++)s=r[c],a=t[s],n(a)||(t[s]=e(a,f?s+o:o,i,s))}function c(e,n,r){try{t.emit(e,n,r)}catch(o){u([o,e,n,r])}}function f(t,e){if(Object.defineProperty&&Object.keys)try{var n=Object.keys(t);return n.forEach(function(n){Object.defineProperty(e,n,{get:function(){return t[n]},set:function(e){return t[n]=e,e}})}),e}catch(r){u([r])}for(var o in t)a.call(t,o)&&(e[o]=t[o]);return e}function u(e){try{t.emit("internal-error",e)}catch(n){}}return t||(t=r),e.inPlace=s,e.flag=i,e}},{1:24,ee:"QJf3ax"}]},{},["G9z0Bl",4,12,6,5]);</script><script type="text/javascript">window.NREUM||(NREUM={});NREUM.info={"beacon":"bam.nr-data.net","queueTime":0,"licenseKey":"a2cef8c3d3","agent":"js-agent.newrelic.com/nr-632.min.js","transactionName":"Z11RZxdWW0cEVkYLDV4XdUYLVEFdClsdAAtEWkZQDlJBGgRFQhFMQl1DXFcZQ10AQkFYBFlUVlEXWEJHAA==","userAttributes":"SxpaQDpWQEANUFwWC1NZR1YBFQ9SBFlBB04SUUBsBEdcFl9TUw4RVRQRRhZSR2sLVF8HQAoacl0KWUxZCkBBQB8=","applicationID":"1841284","errorBeacon":"bam.nr-data.net","applicationTime":153}</script>
  <title>
  natcap / invest 
  / source  / src / natcap / invest / seasonal_water_yield / seasonal_water_yield_core.pyx
 &mdash; Bitbucket
</title>
  


<meta id="bb-canon-url" name="bb-canon-url" content="https://bitbucket.org">

<meta name="description" content=""/>
<meta name="bb-view-name" content="bitbucket.apps.repo2.views.filebrowse">
<meta name="ignore-whitespace" content="False">
<meta name="tab-size" content="None">

<meta name="application-name" content="Bitbucket">
<meta name="apple-mobile-web-app-title" content="Bitbucket">
<meta name="theme-color" content="#205081">
<meta name="msapplication-TileColor" content="#205081">
<meta name="msapplication-TileImage" content="https://d3oaxc4q5k2d6q.cloudfront.net/m/4a1a7b368c96/img/logos/bitbucket/white-256.png">
<link rel="apple-touch-icon" sizes="192x192" type="image/png" href="https://d3oaxc4q5k2d6q.cloudfront.net/m/4a1a7b368c96//img/bitbucket_avatar/192/bitbucket.png">
<link rel="icon" sizes="192x192" type="image/png" href="https://d3oaxc4q5k2d6q.cloudfront.net/m/4a1a7b368c96//img/bitbucket_avatar/192/bitbucket.png">
<link rel="icon" sizes="16x16 32x32" type="image/x-icon" href="/favicon.ico">
<link rel="search" type="application/opensearchdescription+xml" href="/opensearch.xml" title="Bitbucket">

  
    
  
<link rel="stylesheet" href="https://d3oaxc4q5k2d6q.cloudfront.net/m/4a1a7b368c96/compressed/css/9f90c2c1aa07.css" type="text/css" />


  <link rel="stylesheet" href="https://d3oaxc4q5k2d6q.cloudfront.net/m/4a1a7b368c96/compressed/css/770009efab15.css" type="text/css" />


<link rel="stylesheet" href="https://d3oaxc4q5k2d6q.cloudfront.net/m/4a1a7b368c96/compressed/css/08d298fe447a.css" type="text/css" />


  
  <!--[if lte IE 9]><link rel="stylesheet" href="https://d3oaxc4q5k2d6q.cloudfront.net/m/4a1a7b368c96/bower_components/fd-slider/css/fd-slider.css" media="all"><![endif]-->
  <!--[if IE 9]><link rel="stylesheet" href="https://d3oaxc4q5k2d6q.cloudfront.net/m/4a1a7b368c96/bower_components/aui-stable/aui-next/css/aui-ie9.css" media="all"><![endif]-->
  <!--[if IE]><link rel="stylesheet" href="https://d3oaxc4q5k2d6q.cloudfront.net/m/4a1a7b368c96/css/aui-overrides-ie.css" media="all"><![endif]-->
  
  
    <link href="/natcap/invest/rss" rel="alternate nofollow" type="application/rss+xml" title="RSS feed for invest" />
  

</head>
<body class="production aui-page-sidebar aui-sidebar-expanded"
      data-base-url="https://bitbucket.org"
      data-no-avatar-image="https://d3oaxc4q5k2d6q.cloudfront.net/m/4a1a7b368c96/img/default_avatar/16/user_blue.png"
      data-current-user="{&quot;isKbdShortcutsEnabled&quot;: true, &quot;isSshEnabled&quot;: false, &quot;isAuthenticated&quot;: false}"
      data-atlassian-id="{&quot;loginStatusUrl&quot;: &quot;https://id.atlassian.com/profile/rest/profile&quot;}"
      data-settings="{&quot;MENTIONS_MIN_QUERY_LENGTH&quot;: 3}"
      data-flag-super-touch-point="true"
       data-current-repo="{&quot;scm&quot;: &quot;hg&quot;, &quot;creator&quot;: {&quot;username&quot;: &quot;jdouglass&quot;}, &quot;readOnly&quot;: false, &quot;owner&quot;: {&quot;username&quot;: &quot;natcap&quot;, &quot;isTeam&quot;: true}, &quot;slug&quot;: &quot;invest&quot;, &quot;fullslug&quot;: &quot;natcap/invest&quot;, &quot;language&quot;: &quot;python&quot;, &quot;id&quot;: 10013696, &quot;analyticsKey&quot;: &quot;UA-24933546-3&quot;, &quot;mainbranch&quot;: {&quot;name&quot;: &quot;develop&quot;}, &quot;pygmentsLanguage&quot;: &quot;python&quot;}"
       data-current-cset="0fbfb6cecdbdfb95b58f06d9fced7932fd9ec89b"
      
      
      
      
      >

  
    <script type="text/javascript" src="https://d3oaxc4q5k2d6q.cloudfront.net/m/4a1a7b368c96/compressed/js/9f0feb819aab.js"></script>
  


<div id="page">
  <div id="wrapper">
    
  


    <header id="header" role="banner" data-modules="header/tracking">
      
  
  


      <nav class="aui-header aui-dropdown2-trigger-group" role="navigation">
        <div class="aui-header-inner">
          <div class="aui-header-primary">
            
  

            
              <h1 class="aui-header-logo aui-header-logo-bitbucket logged-out" id="logo">
                <a href="/">
                  <span class="aui-header-logo-device">Bitbucket</span>
                </a>
              </h1>
            
            
  
<script id="repo-dropdown-template" type="text/html">
    

[[#hasViewed]]
  <div class="aui-dropdown2-section">
    <strong class="viewed">Recently viewed</strong>
    <ul class="aui-list-truncate">
      [[#viewed]]
        <li class="[[#is_private]]private[[/is_private]][[^is_private]]public[[/is_private]] repository">
          <a href="[[url]]" title="[[owner]]/[[name]]" class="aui-icon-container recently-viewed repo-link">
            <img class="repo-avatar size16" src="[[{avatar}]]" alt="[[owner]]/[[name]] avatar"/>
            [[owner]] / [[name]]
          </a>
        </li>
      [[/viewed]]
    </ul>
  </div>
[[/hasViewed]]
[[#hasUpdated]]
<div class="aui-dropdown2-section">
  <strong class="updated">Recently updated</strong>
  <ul class="aui-list-truncate">
    [[#updated]]
    <li class="[[#is_private]]private[[/is_private]][[^is_private]]public[[/is_private]] repository">
      <a href="[[url]]" title="[[owner]]/[[name]]" class="aui-icon-container recently-updated repo-link">
        <img class="repo-avatar size16" src="[[{avatar}]]" alt="[[owner]]/[[name]] avatar"/>
        [[owner]] / [[name]]
      </a>
    </li>
    [[/updated]]
  </ul>
</div>
[[/hasUpdated]]

  </script>
<script id="snippet-dropdown-template" type="text/html">
    <div class="aui-dropdown2-section">
  <strong>[[sectionTitle]]</strong>
  <ul class="aui-list-truncate">
    [[#snippets]]
      <li>
        <a href="[[links.html.href]]">[[owner.display_name]] / [[name]]</a>
      </li>
    [[/snippets]]
  </ul>
</div>

  </script>
<ul class="aui-nav">
  
    <li>
      <a href="/features">
        Features
      </a>
    </li>
    <li>
      <a href="/plans">
          Pricing
      </a>
    </li>
  
</ul>

          </div>
          <div class="aui-header-secondary">
            
  

<ul role="menu" class="aui-nav">
  
  <li>
    <form action="/repo/all" method="get" class="aui-quicksearch">
      <label for="search-query" class="assistive">owner/repository</label>
      <input id="search-query" class="bb-repo-typeahead" type="text"
             placeholder="Find a repository&hellip;" name="name" autocomplete="off"
             data-bb-typeahead-focus="false">
    </form>
  </li>
  <li id="ace-stp-menu">
    <a id="ace-stp-menu-link" class="aui-nav-link" href="#"
    aria-controls="super-touch-point-dialog"
    
      data-modules="aui/inline-dialog2"
    
    data-aui-trigger>
  <span id="ace-stp-menu-icon"
      class="aui-icon aui-icon-small aui-iconfont-help"></span>
  <div id="ace-stp-menu-icon-notification"></div>
</a>
  </li>
  
    
      <li>
        <a class="aui-dropdown2-trigger" href="#header-language"
            aria-controls="header-language" aria-owns="header-language"
            aria-haspopup="true" data-container="#header .aui-header-inner">
          <span>English</span></a>
        <nav id="header-language" class="aui-dropdown2 aui-style-default aui-dropdown2-radio aui-dropdown2-in-header"
            aria-hidden="true">
          <form method="post" action="/account/language/setlang/"
              data-modules="i18n/header-language-form">
            <input type="hidden" name="language" value="">
            <ul>
            <li><a class="aui-dropdown2-radio interactive checked"
                  data-value="en" href="#en">English</a></li>
            
            <li><a class="aui-dropdown2-radio interactive "
                  data-value="ja" href="#ja">日本語</a></li>
            </ul>
          </form>
        </nav>
      </li>
    
  
  
      <li id="header-signup-button">
        <a id="sign-up-link" class="aui-button aui-button-primary" href="/account/signup/">
          Sign up
        </a>
      </li>
    <li id="user-options">
      <a href="/account/signin/?next=/natcap/invest/src/0fbfb6cecdbdfb95b58f06d9fced7932fd9ec89b/src/natcap/invest/seasonal_water_yield/seasonal_water_yield_core.pyx%3Fat%3Ddevelop" class="aui-nav-link login-link">Log in</a>
    </li>
  
</ul>

          </div>
        </div>
      </nav>
    </header>

    
  <header id="account-warning" role="banner" data-modules="header/account-warning"
        class="aui-message-banner warning
        ">
  <div class="aui-message-banner-inner">
    <span class="aui-icon aui-icon-warning"></span>
    <span class="message">
    
    </span>
  </div>
</header>

    
  
<header id="aui-message-bar">
  
</header>


    <div id="content" role="main">
      
  

<div class="aui-sidebar repo-sidebar" data-modules="components/repo-sidebar,experiment/grow1279-guide"
  >
  <div class="aui-sidebar-wrapper">
    <div class="aui-sidebar-body">
      <header class="aui-page-header">
        <div class="aui-page-header-inner">
          <div class="aui-page-header-image">
            <a href="/natcap/invest" id="repo-avatar-link" class="repo-link">
              <span class="aui-avatar aui-avatar-large aui-avatar-project">
                <span class="aui-avatar-inner">
                  <img  src="https://d3oaxc4q5k2d6q.cloudfront.net/m/4a1a7b368c96/img/language-avatars/default_64.png" class="deferred-image" data-src-url="https://bitbucket.org/natcap/invest/avatar/64/?ts=1436218324" data-src-url-2x="https://bitbucket.org/natcap/invest/avatar/128/?ts=1436218324" alt="">
                </span>
              </span>
            </a>
          </div>
          <div class="aui-page-header-main">
            <ol class="aui-nav aui-nav-breadcrumbs">
              <li>
                <a href="/natcap" id="repo-owner-link">natcap</a>
              </li>
            </ol>
            <h1>
              
              <a href="/natcap/invest" title="invest" class="entity-name">invest</a>
            </h1>
          </div>
        </div>
      </header>
      <nav class="aui-navgroup aui-navgroup-vertical">
        <div class="aui-navgroup-inner">
          
            
              <div class="aui-sidebar-group aui-sidebar-group-actions repository-actions forks-enabled">
                <div class="aui-nav-heading">
                  <strong>Actions</strong>
                </div>
                <ul id="repo-actions" class="aui-nav">
                  
                  
                    <li>
                      <a id="repo-clone-button" class="aui-nav-item "
                        href="#clone"
                        data-modules="components/clone/clone-dialog"
                        target="_self">
                        <span class="aui-icon aui-icon-large icon-clone"></span>
                        <span class="aui-nav-item-label">Clone</span>
                      </a>
                    </li>
                  
                    <li>
                      <a id="repo-compare-link" class="aui-nav-item "
                        href="/natcap/invest/branches/compare"
                        
                        target="_self">
                        <span class="aui-icon aui-icon-large aui-icon-small aui-iconfont-devtools-compare"></span>
                        <span class="aui-nav-item-label">Compare</span>
                      </a>
                    </li>
                  
                    <li>
                      <a id="repo-fork-link" class="aui-nav-item "
                        href="/natcap/invest/fork"
                        
                        target="_self">
                        <span class="aui-icon aui-icon-large icon-fork"></span>
                        <span class="aui-nav-item-label">Fork</span>
                      </a>
                    </li>
                  
                </ul>
              </div>
          
          <div class="aui-sidebar-group aui-sidebar-group-tier-one repository-sections">
            <div class="aui-nav-heading">
              <strong>Navigation</strong>
            </div>
            <ul class="aui-nav">
              
              
                <li>
                  <a id="repo-overview-link" class="aui-nav-item "
                    href="/natcap/invest/overview"
                    
                    target="_self"
                    >
                    
                    <span class="aui-icon aui-icon-large icon-overview"></span>
                    <span class="aui-nav-item-label">Overview</span>
                  </a>
                </li>
              
                <li class="aui-nav-selected">
                  <a id="repo-source-link" class="aui-nav-item "
                    href="/natcap/invest/src"
                    
                    target="_self"
                    >
                    
                    <span class="aui-icon aui-icon-large icon-source"></span>
                    <span class="aui-nav-item-label">Source</span>
                  </a>
                </li>
              
                <li>
                  <a id="repo-commits-link" class="aui-nav-item "
                    href="/natcap/invest/commits"
                    
                    target="_self"
                    >
                    
                    <span class="aui-icon aui-icon-large icon-commits"></span>
                    <span class="aui-nav-item-label">Commits</span>
                  </a>
                </li>
              
                <li>
                  <a id="repo-branches-link" class="aui-nav-item "
                    href="/natcap/invest/branches"
                    
                    target="_self"
                    >
                    
                    <span class="aui-icon aui-icon-large icon-branches"></span>
                    <span class="aui-nav-item-label">Branches</span>
                  </a>
                </li>
              
                <li>
                  <a id="repo-pullrequests-link" class="aui-nav-item "
                    href="/natcap/invest/pull-requests"
                    
                    target="_self"
                    >
                    
                    <span class="aui-icon aui-icon-large icon-pull-requests"></span>
                    <span class="aui-nav-item-label">Pull requests</span>
                  </a>
                </li>
              
                <li>
                  <a id="repo-issues-link" class="aui-nav-item "
                    href="/natcap/invest/issues?status=new&amp;status=open"
                    
                    target="_self"
                    title="( type &#39;r&#39; then &#39;i&#39; )">
                    
                      <span class="aui-badge" title="95 active issues" id="issues-count">95</span>
                    
                    <span class="aui-icon aui-icon-large icon-issues"></span>
                    <span class="aui-nav-item-label">Issues</span>
                  </a>
                </li>
              
                <li>
                  <a id="repo-wiki-link" class="aui-nav-item "
                    href="/natcap/invest/wiki"
                    
                    target="_self"
                    >
                    
                    <span class="aui-icon aui-icon-large icon-wiki"></span>
                    <span class="aui-nav-item-label">Wiki</span>
                  </a>
                </li>
              
                <li>
                  <a id="repo-downloads-link" class="aui-nav-item "
                    href="/natcap/invest/downloads"
                    
                    target="_self"
                    >
                    
                    <span class="aui-icon aui-icon-large icon-downloads"></span>
                    <span class="aui-nav-item-label">Downloads</span>
                  </a>
                </li>
              
            </ul>
          </div>
          <div class="aui-sidebar-group aui-sidebar-group-tier-one repository-settings">
            <div class="aui-nav-heading">
              <strong class="assistive">Settings</strong>
            </div>
            <ul class="aui-nav">
              
              
            </ul>
          </div>
          
            
              <div class="hidden kb-shortcut-actions">
                <a id="repo-create-issue" href="/natcap/invest/issues/new"></a>
              </div>
            
          
        </div>
      </nav>
    </div>
    <div class="aui-sidebar-footer">
      <a class="aui-sidebar-toggle aui-sidebar-footer-tipsy aui-button aui-button-subtle"><span class="aui-icon"></span></a>
    </div>
  </div>
  
  <div id="repo-clone-dialog" class="clone-dialog hidden">
  
  
  <div class="clone-url" data-modules="components/clone/url-dropdown">
  <div class="aui-buttons">
    <a href="https://bitbucket.org/natcap/invest"
      class="aui-button aui-dropdown2-trigger" aria-haspopup="true"
      aria-owns="clone-url-dropdown-header">
      <span class="dropdown-text">HTTPS</span>
    </a>
    <div id="clone-url-dropdown-header"
        class="clone-url-dropdown aui-dropdown2 aui-style-default"
        data-aui-alignment="bottom left">
      <ul class="aui-list-truncate">
        <li>
          <a href="https://bitbucket.org/natcap/invest"
            
              data-command="hg clone https://bitbucket.org/natcap/invest"
            
            class="item-link https">HTTPS
          </a>
        </li>
        <li>
          <a href="ssh://hg@bitbucket.org/natcap/invest"
            
              data-command="hg clone ssh://hg@bitbucket.org/natcap/invest"
            
            class="item-link ssh">SSH
          </a>
        </li>
      </ul>
    </div>
    <input type="text" readonly="readonly" class="clone-url-input"
      value="hg clone https://bitbucket.org/natcap/invest">
  </div>
  
  <p>Need help cloning? Visit
    <a href="https://confluence.atlassian.com/x/cgozDQ" target="_blank">Bitbucket 101</a>.</p>
  
</div>
  
  <div class="sourcetree-callout clone-in-sourcetree"
  data-modules="components/clone/clone-in-sourcetree"
  data-https-url="https://bitbucket.org/natcap/invest"
  data-ssh-url="ssh://hg@bitbucket.org/natcap/invest">

  <div>
    <button class="aui-button aui-button-primary">
      
        Clone in SourceTree
      
    </button>
  </div>

  <p class="windows-text">
    
      <a href="http://www.sourcetreeapp.com/?utm_source=internal&amp;utm_medium=link&amp;utm_campaign=clone_repo_win" target="_blank">Atlassian SourceTree</a>
      is a free Git and Mercurial client for Windows.
    
  </p>
  <p class="mac-text">
    
      <a href="http://www.sourcetreeapp.com/?utm_source=internal&amp;utm_medium=link&amp;utm_campaign=clone_repo_mac" target="_blank">Atlassian SourceTree</a>
      is a free Git and Mercurial client for Mac.
    
  </p>
</div>
</div>
</div>

      
  <div class="aui-page-panel ">
    



    <div class="aui-page-panel-inner">
      <div id="repo-content" class="aui-page-panel-content" data-modules="repo/index">
        <div class="aui-group repo-page-header">
          <div class="aui-item section-title">
            <h1>Source</h1>
          </div>
          <div class="aui-item page-actions">
            
          </div>
        </div>
        
  <div id="source-container" class="maskable" data-modules="repo/source/index">
    



<header id="source-path">
  <div class="labels labels-csv">
    
      <div class="aui-buttons">
        <button data-branches-tags-url="/api/1.0/repositories/natcap/invest/branches-tags"
                data-modules="components/branch-dialog"
                class="aui-button branch-dialog-trigger" title="develop">
          
            
              <span class="aui-icon aui-icon-small aui-iconfont-devtools-branch">Branch</span>
            
            <span class="name">develop</span>
          
          <span class="aui-icon-dropdown"></span>
        </button>
        <button class="aui-button" id="checkout-branch-button"
                title="Check out this branch">
          <span class="aui-icon aui-icon-small aui-iconfont-devtools-clone">Check out branch</span>
          <span class="aui-icon-dropdown"></span>
        </button>
      </div>
      <script id="branch-checkout-template" type="text/html">
  

<div id="checkout-branch-contents">
  <div class="command-line">
    <p>
      Check out this branch on your local machine to begin working on it.
    </p>
    <input type="text" class="checkout-command" readonly="readonly"
        
          value="hg pull && hg update [[branchName]]"
        
        >
  </div>
  
    <div class="sourcetree-callout clone-in-sourcetree"
  data-modules="components/clone/clone-in-sourcetree"
  data-https-url="https://bitbucket.org/natcap/invest"
  data-ssh-url="ssh://hg@bitbucket.org/natcap/invest">

  <div>
    <button class="aui-button aui-button-primary">
      
        Check out in SourceTree
      
    </button>
  </div>

  <p class="windows-text">
    
      <a href="http://www.sourcetreeapp.com/?utm_source=internal&amp;utm_medium=link&amp;utm_campaign=clone_repo_win" target="_blank">Atlassian SourceTree</a>
      is a free Git and Mercurial client for Windows.
    
  </p>
  <p class="mac-text">
    
      <a href="http://www.sourcetreeapp.com/?utm_source=internal&amp;utm_medium=link&amp;utm_campaign=clone_repo_mac" target="_blank">Atlassian SourceTree</a>
      is a free Git and Mercurial client for Mac.
    
  </p>
</div>
  
</div>

</script>
    
  </div>
  <div class="secondary-actions">
    <div class="aui-buttons">
      
        <a href="/natcap/invest/src/0fbfb6cecdbd/src/natcap/invest/seasonal_water_yield/seasonal_water_yield_core.pyx?at=develop"
           class="aui-button pjax-trigger" aria-pressed="true">
          Source
        </a>
        <a href="/natcap/invest/diff/src/natcap/invest/seasonal_water_yield/seasonal_water_yield_core.pyx?diff2=0fbfb6cecdbd&at=develop"
           class="aui-button pjax-trigger"
           title="Diff to previous change">
          Diff
        </a>
        <a href="/natcap/invest/history-node/0fbfb6cecdbd/src/natcap/invest/seasonal_water_yield/seasonal_water_yield_core.pyx?at=develop"
           class="aui-button pjax-trigger">
          History
        </a>
      
    </div>
  </div>
  <h1>
    
      
        <a href="/natcap/invest/src/0fbfb6cecdbd?at=develop"
          class="pjax-trigger root" title="natcap/invest at 0fbfb6cecdbd">invest</a> /
      
      
        
          
            
              <a href="/natcap/invest/src/0fbfb6cecdbd/src/?at=develop"
                class="pjax-trigger directory-name">src</a> /
            
          
        
      
        
          
            
              <a href="/natcap/invest/src/0fbfb6cecdbd/src/natcap/?at=develop"
                class="pjax-trigger directory-name">natcap</a> /
            
          
        
      
        
          
            
              <a href="/natcap/invest/src/0fbfb6cecdbd/src/natcap/invest/?at=develop"
                class="pjax-trigger directory-name">invest</a> /
            
          
        
      
        
          
            
              <a href="/natcap/invest/src/0fbfb6cecdbd/src/natcap/invest/seasonal_water_yield/?at=develop"
                class="pjax-trigger directory-name">seasonal_water_yield</a> /
            
          
        
      
        
          
            
              <span class="file-name">seasonal_water_yield_core.pyx</span>
            
          
        
      
    
  </h1>
  
    
    
  
  <div class="clearfix"></div>
</header>


    
      
    

    <div id="editor-container" class="maskable"
         data-modules="repo/source/editor"
         data-owner="natcap"
         data-slug="invest"
         data-is-writer="false"
         data-has-push-access="true"
         data-hash="0fbfb6cecdbdfb95b58f06d9fced7932fd9ec89b"
         data-branch="develop"
         data-path="src/natcap/invest/seasonal_water_yield/seasonal_water_yield_core.pyx"
         data-source-url="/api/1.0/repositories/natcap/invest/src/0fbfb6cecdbdfb95b58f06d9fced7932fd9ec89b/src/natcap/invest/seasonal_water_yield/seasonal_water_yield_core.pyx">
      <div id="source-view" class="file-source-container" data-modules="repo/source/view-file">
        <div class="toolbar">
          <div class="primary">
            <div class="aui-buttons">
              
                <button id="file-history-trigger" class="aui-button aui-button-light changeset-info"
                        data-changeset="0fbfb6cecdbdfb95b58f06d9fced7932fd9ec89b"
                        data-path="src/natcap/invest/seasonal_water_yield/seasonal_water_yield_core.pyx"
                        data-current="0fbfb6cecdbdfb95b58f06d9fced7932fd9ec89b">
                  
                     

  <div class="aui-avatar aui-avatar-xsmall">
    <div class="aui-avatar-inner">
      <img src="https://bitbucket.org/account/richpsharp/avatar/16/?ts=0">
    </div>
  </div>
  <span class="changeset-hash">0fbfb6c</span>
  <time datetime="2015-07-06T04:05:24+00:00" class="timestamp"></time>
  <span class="aui-icon-dropdown"></span>

                  
                </button>
              
            </div>
          <a href="/natcap/invest/full-commit/0fbfb6cecdbd/src/natcap/invest/seasonal_water_yield/seasonal_water_yield_core.pyx" id="full-commit-link"
              title="View full commit 0fbfb6c">Full commit</a>
          </div>
            <div class="secondary">
              <div class="aui-buttons">
                
                  <a href="/natcap/invest/annotate/0fbfb6cecdbdfb95b58f06d9fced7932fd9ec89b/src/natcap/invest/seasonal_water_yield/seasonal_water_yield_core.pyx?at=develop"
                  class="aui-button aui-button-light pjax-trigger">Blame</a>
                
                
                  
                  <a id="embed-link" href="https://bitbucket.org/natcap/invest/src/0fbfb6cecdbdfb95b58f06d9fced7932fd9ec89b/src/natcap/invest/seasonal_water_yield/seasonal_water_yield_core.pyx?embed=t"
                    class="aui-button aui-button-light" data-modules="repo/source/embed">Embed</a>
                
                <a href="/natcap/invest/raw/0fbfb6cecdbdfb95b58f06d9fced7932fd9ec89b/src/natcap/invest/seasonal_water_yield/seasonal_water_yield_core.pyx"
                  class="aui-button aui-button-light">Raw</a>
              </div>
              
                <button class="edit-button aui-button aui-button-light" disabled="disabled" aria-disabled="true">
                  Edit
                  <span class="edit-button-overlay" title="Log in to edit this file"></span>
                </button>
              
            </div>
          <div class="clearfix"></div>
        </div>
        


  <div class="file-source">
    <table class="highlighttable">
<tr><td class="linenos"><div class="linenodiv"><pre>
<a href="#cl-1">1</a>
<a href="#cl-2">2</a>
<a href="#cl-3">3</a>
<a href="#cl-4">4</a>
<a href="#cl-5">5</a>
<a href="#cl-6">6</a>
<a href="#cl-7">7</a>
<a href="#cl-8">8</a>
<a href="#cl-9">9</a>
<a href="#cl-10">10</a>
<a href="#cl-11">11</a>
<a href="#cl-12">12</a>
<a href="#cl-13">13</a>
<a href="#cl-14">14</a>
<a href="#cl-15">15</a>
<a href="#cl-16">16</a>
<a href="#cl-17">17</a>
<a href="#cl-18">18</a>
<a href="#cl-19">19</a>
<a href="#cl-20">20</a>
<a href="#cl-21">21</a>
<a href="#cl-22">22</a>
<a href="#cl-23">23</a>
<a href="#cl-24">24</a>
<a href="#cl-25">25</a>
<a href="#cl-26">26</a>
<a href="#cl-27">27</a>
<a href="#cl-28">28</a>
<a href="#cl-29">29</a>
<a href="#cl-30">30</a>
<a href="#cl-31">31</a>
<a href="#cl-32">32</a>
<a href="#cl-33">33</a>
<a href="#cl-34">34</a>
<a href="#cl-35">35</a>
<a href="#cl-36">36</a>
<a href="#cl-37">37</a>
<a href="#cl-38">38</a>
<a href="#cl-39">39</a>
<a href="#cl-40">40</a>
<a href="#cl-41">41</a>
<a href="#cl-42">42</a>
<a href="#cl-43">43</a>
<a href="#cl-44">44</a>
<a href="#cl-45">45</a>
<a href="#cl-46">46</a>
<a href="#cl-47">47</a>
<a href="#cl-48">48</a>
<a href="#cl-49">49</a>
<a href="#cl-50">50</a>
<a href="#cl-51">51</a>
<a href="#cl-52">52</a>
<a href="#cl-53">53</a>
<a href="#cl-54">54</a>
<a href="#cl-55">55</a>
<a href="#cl-56">56</a>
<a href="#cl-57">57</a>
<a href="#cl-58">58</a>
<a href="#cl-59">59</a>
<a href="#cl-60">60</a>
<a href="#cl-61">61</a>
<a href="#cl-62">62</a>
<a href="#cl-63">63</a>
<a href="#cl-64">64</a>
<a href="#cl-65">65</a>
<a href="#cl-66">66</a>
<a href="#cl-67">67</a>
<a href="#cl-68">68</a>
<a href="#cl-69">69</a>
<a href="#cl-70">70</a>
<a href="#cl-71">71</a>
<a href="#cl-72">72</a>
<a href="#cl-73">73</a>
<a href="#cl-74">74</a>
<a href="#cl-75">75</a>
<a href="#cl-76">76</a>
<a href="#cl-77">77</a>
<a href="#cl-78">78</a>
<a href="#cl-79">79</a>
<a href="#cl-80">80</a>
<a href="#cl-81">81</a>
<a href="#cl-82">82</a>
<a href="#cl-83">83</a>
<a href="#cl-84">84</a>
<a href="#cl-85">85</a>
<a href="#cl-86">86</a>
<a href="#cl-87">87</a>
<a href="#cl-88">88</a>
<a href="#cl-89">89</a>
<a href="#cl-90">90</a>
<a href="#cl-91">91</a>
<a href="#cl-92">92</a>
<a href="#cl-93">93</a>
<a href="#cl-94">94</a>
<a href="#cl-95">95</a>
<a href="#cl-96">96</a>
<a href="#cl-97">97</a>
<a href="#cl-98">98</a>
<a href="#cl-99">99</a>
<a href="#cl-100">100</a>
<a href="#cl-101">101</a>
<a href="#cl-102">102</a>
<a href="#cl-103">103</a>
<a href="#cl-104">104</a>
<a href="#cl-105">105</a>
<a href="#cl-106">106</a>
<a href="#cl-107">107</a>
<a href="#cl-108">108</a>
<a href="#cl-109">109</a>
<a href="#cl-110">110</a>
<a href="#cl-111">111</a>
<a href="#cl-112">112</a>
<a href="#cl-113">113</a>
<a href="#cl-114">114</a>
<a href="#cl-115">115</a>
<a href="#cl-116">116</a>
<a href="#cl-117">117</a>
<a href="#cl-118">118</a>
<a href="#cl-119">119</a>
<a href="#cl-120">120</a>
<a href="#cl-121">121</a>
<a href="#cl-122">122</a>
<a href="#cl-123">123</a>
<a href="#cl-124">124</a>
<a href="#cl-125">125</a>
<a href="#cl-126">126</a>
<a href="#cl-127">127</a>
<a href="#cl-128">128</a>
<a href="#cl-129">129</a>
<a href="#cl-130">130</a>
<a href="#cl-131">131</a>
<a href="#cl-132">132</a>
<a href="#cl-133">133</a>
<a href="#cl-134">134</a>
<a href="#cl-135">135</a>
<a href="#cl-136">136</a>
<a href="#cl-137">137</a>
<a href="#cl-138">138</a>
<a href="#cl-139">139</a>
<a href="#cl-140">140</a>
<a href="#cl-141">141</a>
<a href="#cl-142">142</a>
<a href="#cl-143">143</a>
<a href="#cl-144">144</a>
<a href="#cl-145">145</a>
<a href="#cl-146">146</a>
<a href="#cl-147">147</a>
<a href="#cl-148">148</a>
<a href="#cl-149">149</a>
<a href="#cl-150">150</a>
<a href="#cl-151">151</a>
<a href="#cl-152">152</a>
<a href="#cl-153">153</a>
<a href="#cl-154">154</a>
<a href="#cl-155">155</a>
<a href="#cl-156">156</a>
<a href="#cl-157">157</a>
<a href="#cl-158">158</a>
<a href="#cl-159">159</a>
<a href="#cl-160">160</a>
<a href="#cl-161">161</a>
<a href="#cl-162">162</a>
<a href="#cl-163">163</a>
<a href="#cl-164">164</a>
<a href="#cl-165">165</a>
<a href="#cl-166">166</a>
<a href="#cl-167">167</a>
<a href="#cl-168">168</a>
<a href="#cl-169">169</a>
<a href="#cl-170">170</a>
<a href="#cl-171">171</a>
<a href="#cl-172">172</a>
<a href="#cl-173">173</a>
<a href="#cl-174">174</a>
<a href="#cl-175">175</a>
<a href="#cl-176">176</a>
<a href="#cl-177">177</a>
<a href="#cl-178">178</a>
<a href="#cl-179">179</a>
<a href="#cl-180">180</a>
<a href="#cl-181">181</a>
<a href="#cl-182">182</a>
<a href="#cl-183">183</a>
<a href="#cl-184">184</a>
<a href="#cl-185">185</a>
<a href="#cl-186">186</a>
<a href="#cl-187">187</a>
<a href="#cl-188">188</a>
<a href="#cl-189">189</a>
<a href="#cl-190">190</a>
<a href="#cl-191">191</a>
<a href="#cl-192">192</a>
<a href="#cl-193">193</a>
<a href="#cl-194">194</a>
<a href="#cl-195">195</a>
<a href="#cl-196">196</a>
<a href="#cl-197">197</a>
<a href="#cl-198">198</a>
<a href="#cl-199">199</a>
<a href="#cl-200">200</a>
<a href="#cl-201">201</a>
<a href="#cl-202">202</a>
<a href="#cl-203">203</a>
<a href="#cl-204">204</a>
<a href="#cl-205">205</a>
<a href="#cl-206">206</a>
<a href="#cl-207">207</a>
<a href="#cl-208">208</a>
<a href="#cl-209">209</a>
<a href="#cl-210">210</a>
<a href="#cl-211">211</a>
<a href="#cl-212">212</a>
<a href="#cl-213">213</a>
<a href="#cl-214">214</a>
<a href="#cl-215">215</a>
<a href="#cl-216">216</a>
<a href="#cl-217">217</a>
<a href="#cl-218">218</a>
<a href="#cl-219">219</a>
<a href="#cl-220">220</a>
<a href="#cl-221">221</a>
<a href="#cl-222">222</a>
<a href="#cl-223">223</a>
<a href="#cl-224">224</a>
<a href="#cl-225">225</a>
<a href="#cl-226">226</a>
<a href="#cl-227">227</a>
<a href="#cl-228">228</a>
<a href="#cl-229">229</a>
<a href="#cl-230">230</a>
<a href="#cl-231">231</a>
<a href="#cl-232">232</a>
<a href="#cl-233">233</a>
<a href="#cl-234">234</a>
<a href="#cl-235">235</a>
<a href="#cl-236">236</a>
<a href="#cl-237">237</a>
<a href="#cl-238">238</a>
<a href="#cl-239">239</a>
<a href="#cl-240">240</a>
<a href="#cl-241">241</a>
<a href="#cl-242">242</a>
<a href="#cl-243">243</a>
<a href="#cl-244">244</a>
<a href="#cl-245">245</a>
<a href="#cl-246">246</a>
<a href="#cl-247">247</a>
<a href="#cl-248">248</a>
<a href="#cl-249">249</a>
<a href="#cl-250">250</a>
<a href="#cl-251">251</a>
<a href="#cl-252">252</a>
<a href="#cl-253">253</a>
<a href="#cl-254">254</a>
<a href="#cl-255">255</a>
<a href="#cl-256">256</a>
<a href="#cl-257">257</a>
<a href="#cl-258">258</a>
<a href="#cl-259">259</a>
<a href="#cl-260">260</a>
<a href="#cl-261">261</a>
<a href="#cl-262">262</a>
<a href="#cl-263">263</a>
<a href="#cl-264">264</a>
<a href="#cl-265">265</a>
<a href="#cl-266">266</a>
<a href="#cl-267">267</a>
<a href="#cl-268">268</a>
<a href="#cl-269">269</a>
<a href="#cl-270">270</a>
<a href="#cl-271">271</a>
<a href="#cl-272">272</a>
<a href="#cl-273">273</a>
<a href="#cl-274">274</a>
<a href="#cl-275">275</a>
<a href="#cl-276">276</a>
<a href="#cl-277">277</a>
<a href="#cl-278">278</a>
<a href="#cl-279">279</a>
<a href="#cl-280">280</a>
<a href="#cl-281">281</a>
<a href="#cl-282">282</a>
<a href="#cl-283">283</a>
<a href="#cl-284">284</a>
<a href="#cl-285">285</a>
<a href="#cl-286">286</a>
<a href="#cl-287">287</a>
<a href="#cl-288">288</a>
<a href="#cl-289">289</a>
<a href="#cl-290">290</a>
<a href="#cl-291">291</a>
<a href="#cl-292">292</a>
<a href="#cl-293">293</a>
<a href="#cl-294">294</a>
<a href="#cl-295">295</a>
<a href="#cl-296">296</a>
<a href="#cl-297">297</a>
<a href="#cl-298">298</a>
<a href="#cl-299">299</a>
<a href="#cl-300">300</a>
<a href="#cl-301">301</a>
<a href="#cl-302">302</a>
<a href="#cl-303">303</a>
<a href="#cl-304">304</a>
<a href="#cl-305">305</a>
<a href="#cl-306">306</a>
<a href="#cl-307">307</a>
<a href="#cl-308">308</a>
<a href="#cl-309">309</a>
<a href="#cl-310">310</a>
<a href="#cl-311">311</a>
<a href="#cl-312">312</a>
<a href="#cl-313">313</a>
<a href="#cl-314">314</a>
<a href="#cl-315">315</a>
<a href="#cl-316">316</a>
<a href="#cl-317">317</a>
<a href="#cl-318">318</a>
<a href="#cl-319">319</a>
<a href="#cl-320">320</a>
<a href="#cl-321">321</a>
<a href="#cl-322">322</a>
<a href="#cl-323">323</a>
<a href="#cl-324">324</a>
<a href="#cl-325">325</a>
<a href="#cl-326">326</a>
<a href="#cl-327">327</a>
<a href="#cl-328">328</a>
<a href="#cl-329">329</a>
<a href="#cl-330">330</a>
<a href="#cl-331">331</a>
<a href="#cl-332">332</a>
<a href="#cl-333">333</a>
<a href="#cl-334">334</a>
<a href="#cl-335">335</a>
<a href="#cl-336">336</a>
<a href="#cl-337">337</a>
<a href="#cl-338">338</a>
<a href="#cl-339">339</a>
<a href="#cl-340">340</a>
<a href="#cl-341">341</a>
<a href="#cl-342">342</a>
<a href="#cl-343">343</a>
<a href="#cl-344">344</a>
<a href="#cl-345">345</a>
<a href="#cl-346">346</a>
<a href="#cl-347">347</a>
<a href="#cl-348">348</a>
<a href="#cl-349">349</a>
<a href="#cl-350">350</a>
<a href="#cl-351">351</a>
<a href="#cl-352">352</a>
<a href="#cl-353">353</a>
<a href="#cl-354">354</a>
<a href="#cl-355">355</a>
<a href="#cl-356">356</a>
<a href="#cl-357">357</a>
<a href="#cl-358">358</a>
<a href="#cl-359">359</a>
<a href="#cl-360">360</a>
<a href="#cl-361">361</a>
<a href="#cl-362">362</a>
<a href="#cl-363">363</a>
<a href="#cl-364">364</a>
<a href="#cl-365">365</a>
<a href="#cl-366">366</a>
<a href="#cl-367">367</a>
<a href="#cl-368">368</a>
<a href="#cl-369">369</a>
<a href="#cl-370">370</a>
<a href="#cl-371">371</a>
<a href="#cl-372">372</a>
<a href="#cl-373">373</a>
<a href="#cl-374">374</a>
<a href="#cl-375">375</a>
<a href="#cl-376">376</a>
<a href="#cl-377">377</a>
<a href="#cl-378">378</a>
<a href="#cl-379">379</a>
<a href="#cl-380">380</a>
<a href="#cl-381">381</a>
<a href="#cl-382">382</a>
<a href="#cl-383">383</a>
<a href="#cl-384">384</a>
<a href="#cl-385">385</a>
<a href="#cl-386">386</a>
<a href="#cl-387">387</a>
<a href="#cl-388">388</a>
<a href="#cl-389">389</a>
<a href="#cl-390">390</a>
<a href="#cl-391">391</a>
<a href="#cl-392">392</a>
<a href="#cl-393">393</a>
<a href="#cl-394">394</a>
<a href="#cl-395">395</a>
<a href="#cl-396">396</a>
<a href="#cl-397">397</a>
<a href="#cl-398">398</a>
<a href="#cl-399">399</a>
<a href="#cl-400">400</a>
<a href="#cl-401">401</a>
<a href="#cl-402">402</a>
<a href="#cl-403">403</a>
<a href="#cl-404">404</a>
<a href="#cl-405">405</a>
<a href="#cl-406">406</a>
<a href="#cl-407">407</a>
<a href="#cl-408">408</a>
<a href="#cl-409">409</a>
<a href="#cl-410">410</a>
<a href="#cl-411">411</a>
<a href="#cl-412">412</a>
<a href="#cl-413">413</a>
<a href="#cl-414">414</a>
<a href="#cl-415">415</a>
<a href="#cl-416">416</a>
<a href="#cl-417">417</a>
<a href="#cl-418">418</a>
<a href="#cl-419">419</a>
<a href="#cl-420">420</a>
<a href="#cl-421">421</a>
<a href="#cl-422">422</a>
<a href="#cl-423">423</a>
<a href="#cl-424">424</a>
<a href="#cl-425">425</a>
<a href="#cl-426">426</a>
<a href="#cl-427">427</a>
<a href="#cl-428">428</a>
<a href="#cl-429">429</a>
<a href="#cl-430">430</a>
<a href="#cl-431">431</a>
<a href="#cl-432">432</a>
<a href="#cl-433">433</a>
<a href="#cl-434">434</a>
<a href="#cl-435">435</a>
<a href="#cl-436">436</a>
<a href="#cl-437">437</a>
<a href="#cl-438">438</a>
<a href="#cl-439">439</a>
<a href="#cl-440">440</a>
<a href="#cl-441">441</a>
<a href="#cl-442">442</a>
<a href="#cl-443">443</a>
<a href="#cl-444">444</a>
<a href="#cl-445">445</a>
<a href="#cl-446">446</a>
<a href="#cl-447">447</a>
<a href="#cl-448">448</a>
<a href="#cl-449">449</a>
<a href="#cl-450">450</a>
<a href="#cl-451">451</a>
<a href="#cl-452">452</a>
<a href="#cl-453">453</a>
<a href="#cl-454">454</a>
<a href="#cl-455">455</a>
<a href="#cl-456">456</a>
<a href="#cl-457">457</a>
<a href="#cl-458">458</a>
<a href="#cl-459">459</a>
<a href="#cl-460">460</a>
<a href="#cl-461">461</a>
<a href="#cl-462">462</a>
<a href="#cl-463">463</a>
<a href="#cl-464">464</a>
<a href="#cl-465">465</a>
<a href="#cl-466">466</a>
<a href="#cl-467">467</a>
<a href="#cl-468">468</a>
<a href="#cl-469">469</a>
<a href="#cl-470">470</a>
<a href="#cl-471">471</a>
<a href="#cl-472">472</a>
<a href="#cl-473">473</a>
<a href="#cl-474">474</a>
<a href="#cl-475">475</a>
<a href="#cl-476">476</a>
<a href="#cl-477">477</a>
<a href="#cl-478">478</a>
<a href="#cl-479">479</a>
<a href="#cl-480">480</a>
<a href="#cl-481">481</a>
<a href="#cl-482">482</a>
<a href="#cl-483">483</a>
<a href="#cl-484">484</a>
<a href="#cl-485">485</a>
<a href="#cl-486">486</a>
<a href="#cl-487">487</a>
<a href="#cl-488">488</a>
<a href="#cl-489">489</a>
<a href="#cl-490">490</a>
<a href="#cl-491">491</a>
<a href="#cl-492">492</a>
<a href="#cl-493">493</a>
<a href="#cl-494">494</a>
<a href="#cl-495">495</a>
<a href="#cl-496">496</a>
<a href="#cl-497">497</a>
<a href="#cl-498">498</a>
<a href="#cl-499">499</a>
<a href="#cl-500">500</a>
<a href="#cl-501">501</a>
<a href="#cl-502">502</a>
<a href="#cl-503">503</a>
<a href="#cl-504">504</a>
<a href="#cl-505">505</a>
<a href="#cl-506">506</a>
<a href="#cl-507">507</a>
<a href="#cl-508">508</a>
<a href="#cl-509">509</a>
<a href="#cl-510">510</a>
<a href="#cl-511">511</a>
<a href="#cl-512">512</a>
<a href="#cl-513">513</a>
<a href="#cl-514">514</a>
<a href="#cl-515">515</a>
<a href="#cl-516">516</a>
<a href="#cl-517">517</a>
<a href="#cl-518">518</a>
<a href="#cl-519">519</a>
<a href="#cl-520">520</a>
<a href="#cl-521">521</a>
<a href="#cl-522">522</a>
<a href="#cl-523">523</a>
<a href="#cl-524">524</a>
<a href="#cl-525">525</a>
<a href="#cl-526">526</a>
<a href="#cl-527">527</a>
<a href="#cl-528">528</a>
<a href="#cl-529">529</a>
<a href="#cl-530">530</a>
<a href="#cl-531">531</a>
<a href="#cl-532">532</a>
<a href="#cl-533">533</a>
<a href="#cl-534">534</a>
<a href="#cl-535">535</a>
<a href="#cl-536">536</a>
<a href="#cl-537">537</a>
<a href="#cl-538">538</a>
<a href="#cl-539">539</a>
<a href="#cl-540">540</a>
<a href="#cl-541">541</a>
<a href="#cl-542">542</a>
<a href="#cl-543">543</a>
<a href="#cl-544">544</a>
<a href="#cl-545">545</a>
<a href="#cl-546">546</a>
<a href="#cl-547">547</a>
<a href="#cl-548">548</a>
<a href="#cl-549">549</a>
<a href="#cl-550">550</a>
<a href="#cl-551">551</a>
<a href="#cl-552">552</a>
<a href="#cl-553">553</a>
<a href="#cl-554">554</a>
<a href="#cl-555">555</a>
<a href="#cl-556">556</a>
<a href="#cl-557">557</a>
<a href="#cl-558">558</a>
<a href="#cl-559">559</a>
<a href="#cl-560">560</a>
<a href="#cl-561">561</a>
<a href="#cl-562">562</a>
<a href="#cl-563">563</a>
<a href="#cl-564">564</a>
<a href="#cl-565">565</a>
<a href="#cl-566">566</a>
<a href="#cl-567">567</a>
<a href="#cl-568">568</a>
<a href="#cl-569">569</a>
<a href="#cl-570">570</a>
<a href="#cl-571">571</a>
<a href="#cl-572">572</a>
<a href="#cl-573">573</a>
<a href="#cl-574">574</a>
<a href="#cl-575">575</a>
<a href="#cl-576">576</a>
<a href="#cl-577">577</a>
<a href="#cl-578">578</a>
<a href="#cl-579">579</a>
<a href="#cl-580">580</a>
<a href="#cl-581">581</a>
<a href="#cl-582">582</a>
<a href="#cl-583">583</a>
<a href="#cl-584">584</a>
<a href="#cl-585">585</a>
<a href="#cl-586">586</a>
<a href="#cl-587">587</a>
<a href="#cl-588">588</a>
<a href="#cl-589">589</a>
<a href="#cl-590">590</a>
<a href="#cl-591">591</a>
<a href="#cl-592">592</a>
<a href="#cl-593">593</a>
<a href="#cl-594">594</a>
<a href="#cl-595">595</a>
<a href="#cl-596">596</a>
<a href="#cl-597">597</a>
<a href="#cl-598">598</a>
<a href="#cl-599">599</a>
<a href="#cl-600">600</a>
<a href="#cl-601">601</a>
<a href="#cl-602">602</a>
<a href="#cl-603">603</a>
<a href="#cl-604">604</a>
<a href="#cl-605">605</a>
<a href="#cl-606">606</a>
<a href="#cl-607">607</a>
<a href="#cl-608">608</a>
<a href="#cl-609">609</a>
<a href="#cl-610">610</a>
<a href="#cl-611">611</a>
<a href="#cl-612">612</a>
<a href="#cl-613">613</a>
<a href="#cl-614">614</a>
<a href="#cl-615">615</a>
<a href="#cl-616">616</a>
<a href="#cl-617">617</a>
<a href="#cl-618">618</a>
<a href="#cl-619">619</a>
<a href="#cl-620">620</a>
<a href="#cl-621">621</a>
<a href="#cl-622">622</a>
<a href="#cl-623">623</a>
<a href="#cl-624">624</a>
<a href="#cl-625">625</a>
<a href="#cl-626">626</a>
<a href="#cl-627">627</a>
<a href="#cl-628">628</a>
<a href="#cl-629">629</a>
<a href="#cl-630">630</a>
<a href="#cl-631">631</a>
<a href="#cl-632">632</a>
<a href="#cl-633">633</a>
<a href="#cl-634">634</a>
<a href="#cl-635">635</a>
<a href="#cl-636">636</a>
<a href="#cl-637">637</a>
<a href="#cl-638">638</a>
<a href="#cl-639">639</a>
<a href="#cl-640">640</a>
<a href="#cl-641">641</a>
<a href="#cl-642">642</a>
<a href="#cl-643">643</a>
<a href="#cl-644">644</a>
<a href="#cl-645">645</a>
<a href="#cl-646">646</a>
<a href="#cl-647">647</a>
<a href="#cl-648">648</a>
<a href="#cl-649">649</a>
<a href="#cl-650">650</a>
<a href="#cl-651">651</a>
<a href="#cl-652">652</a>
<a href="#cl-653">653</a>
<a href="#cl-654">654</a>
<a href="#cl-655">655</a>
<a href="#cl-656">656</a>
<a href="#cl-657">657</a>
<a href="#cl-658">658</a>
<a href="#cl-659">659</a>
<a href="#cl-660">660</a>
<a href="#cl-661">661</a>
<a href="#cl-662">662</a>
<a href="#cl-663">663</a>
<a href="#cl-664">664</a>
<a href="#cl-665">665</a>
<a href="#cl-666">666</a>
<a href="#cl-667">667</a>
<a href="#cl-668">668</a>
<a href="#cl-669">669</a>
<a href="#cl-670">670</a>
<a href="#cl-671">671</a>
<a href="#cl-672">672</a>
<a href="#cl-673">673</a>
<a href="#cl-674">674</a>
<a href="#cl-675">675</a>
<a href="#cl-676">676</a>
<a href="#cl-677">677</a>
<a href="#cl-678">678</a>
<a href="#cl-679">679</a>
<a href="#cl-680">680</a>
<a href="#cl-681">681</a>
<a href="#cl-682">682</a>
<a href="#cl-683">683</a>
<a href="#cl-684">684</a>
<a href="#cl-685">685</a>
<a href="#cl-686">686</a>
<a href="#cl-687">687</a>
<a href="#cl-688">688</a>
<a href="#cl-689">689</a>
<a href="#cl-690">690</a>
<a href="#cl-691">691</a>
<a href="#cl-692">692</a>
<a href="#cl-693">693</a>
<a href="#cl-694">694</a>
<a href="#cl-695">695</a>
<a href="#cl-696">696</a>
<a href="#cl-697">697</a>
<a href="#cl-698">698</a>
<a href="#cl-699">699</a>
<a href="#cl-700">700</a>
<a href="#cl-701">701</a>
<a href="#cl-702">702</a>
<a href="#cl-703">703</a>
<a href="#cl-704">704</a>
<a href="#cl-705">705</a>
<a href="#cl-706">706</a>
<a href="#cl-707">707</a>
<a href="#cl-708">708</a>
<a href="#cl-709">709</a>
<a href="#cl-710">710</a>
<a href="#cl-711">711</a>
<a href="#cl-712">712</a>
<a href="#cl-713">713</a>
<a href="#cl-714">714</a>
<a href="#cl-715">715</a>
<a href="#cl-716">716</a>
<a href="#cl-717">717</a>
<a href="#cl-718">718</a>
<a href="#cl-719">719</a>
<a href="#cl-720">720</a>
<a href="#cl-721">721</a>
<a href="#cl-722">722</a>
<a href="#cl-723">723</a>
<a href="#cl-724">724</a>
<a href="#cl-725">725</a>
<a href="#cl-726">726</a>
<a href="#cl-727">727</a>
<a href="#cl-728">728</a>
<a href="#cl-729">729</a>
<a href="#cl-730">730</a>
<a href="#cl-731">731</a>
<a href="#cl-732">732</a>
<a href="#cl-733">733</a>
<a href="#cl-734">734</a>
<a href="#cl-735">735</a>
<a href="#cl-736">736</a>
<a href="#cl-737">737</a>
<a href="#cl-738">738</a>
<a href="#cl-739">739</a>
<a href="#cl-740">740</a>
<a href="#cl-741">741</a>
<a href="#cl-742">742</a>
<a href="#cl-743">743</a>
<a href="#cl-744">744</a>
<a href="#cl-745">745</a>
<a href="#cl-746">746</a>
<a href="#cl-747">747</a>
<a href="#cl-748">748</a>
<a href="#cl-749">749</a>
<a href="#cl-750">750</a>
<a href="#cl-751">751</a>
<a href="#cl-752">752</a>
<a href="#cl-753">753</a>
<a href="#cl-754">754</a>
<a href="#cl-755">755</a>
<a href="#cl-756">756</a>
<a href="#cl-757">757</a>
<a href="#cl-758">758</a>
<a href="#cl-759">759</a>
<a href="#cl-760">760</a>
<a href="#cl-761">761</a>
<a href="#cl-762">762</a>
<a href="#cl-763">763</a>
<a href="#cl-764">764</a>
<a href="#cl-765">765</a>
<a href="#cl-766">766</a>
<a href="#cl-767">767</a>
<a href="#cl-768">768</a>
<a href="#cl-769">769</a>
<a href="#cl-770">770</a>
<a href="#cl-771">771</a>
<a href="#cl-772">772</a>
<a href="#cl-773">773</a>
<a href="#cl-774">774</a>
<a href="#cl-775">775</a>
<a href="#cl-776">776</a>
<a href="#cl-777">777</a>
<a href="#cl-778">778</a>
<a href="#cl-779">779</a>
<a href="#cl-780">780</a>
<a href="#cl-781">781</a>
<a href="#cl-782">782</a>
<a href="#cl-783">783</a>
<a href="#cl-784">784</a>
<a href="#cl-785">785</a>
<a href="#cl-786">786</a>
<a href="#cl-787">787</a>
<a href="#cl-788">788</a>
<a href="#cl-789">789</a>
<a href="#cl-790">790</a>
<a href="#cl-791">791</a>
<a href="#cl-792">792</a>
<a href="#cl-793">793</a>
<a href="#cl-794">794</a>
<a href="#cl-795">795</a>
<a href="#cl-796">796</a>
<a href="#cl-797">797</a>
<a href="#cl-798">798</a>
<a href="#cl-799">799</a>
<a href="#cl-800">800</a>
<a href="#cl-801">801</a>
<a href="#cl-802">802</a>
<a href="#cl-803">803</a>
<a href="#cl-804">804</a>
<a href="#cl-805">805</a>
<a href="#cl-806">806</a>
<a href="#cl-807">807</a>
<a href="#cl-808">808</a>
<a href="#cl-809">809</a>
<a href="#cl-810">810</a>
<a href="#cl-811">811</a>
<a href="#cl-812">812</a>
<a href="#cl-813">813</a>
<a href="#cl-814">814</a>
<a href="#cl-815">815</a>
<a href="#cl-816">816</a>
<a href="#cl-817">817</a>
<a href="#cl-818">818</a>
<a href="#cl-819">819</a>
<a href="#cl-820">820</a>
<a href="#cl-821">821</a>
<a href="#cl-822">822</a>
<a href="#cl-823">823</a>
<a href="#cl-824">824</a>
<a href="#cl-825">825</a>
<a href="#cl-826">826</a>
<a href="#cl-827">827</a>
<a href="#cl-828">828</a>
<a href="#cl-829">829</a>
<a href="#cl-830">830</a>
<a href="#cl-831">831</a>
<a href="#cl-832">832</a>
<a href="#cl-833">833</a>
<a href="#cl-834">834</a>
<a href="#cl-835">835</a>
<a href="#cl-836">836</a>
<a href="#cl-837">837</a>
<a href="#cl-838">838</a>
<a href="#cl-839">839</a>
<a href="#cl-840">840</a>
<a href="#cl-841">841</a>
<a href="#cl-842">842</a>
<a href="#cl-843">843</a>
<a href="#cl-844">844</a>
<a href="#cl-845">845</a>
<a href="#cl-846">846</a>
<a href="#cl-847">847</a>
<a href="#cl-848">848</a>
<a href="#cl-849">849</a>
<a href="#cl-850">850</a>
<a href="#cl-851">851</a>
<a href="#cl-852">852</a>
<a href="#cl-853">853</a>
<a href="#cl-854">854</a>
<a href="#cl-855">855</a>
<a href="#cl-856">856</a>
<a href="#cl-857">857</a>
<a href="#cl-858">858</a>
<a href="#cl-859">859</a>
<a href="#cl-860">860</a>
<a href="#cl-861">861</a>
<a href="#cl-862">862</a>
<a href="#cl-863">863</a>
<a href="#cl-864">864</a>
<a href="#cl-865">865</a>
<a href="#cl-866">866</a>
<a href="#cl-867">867</a>
<a href="#cl-868">868</a>
<a href="#cl-869">869</a>
<a href="#cl-870">870</a>
<a href="#cl-871">871</a>
<a href="#cl-872">872</a>
<a href="#cl-873">873</a>
<a href="#cl-874">874</a>
<a href="#cl-875">875</a>
<a href="#cl-876">876</a>
<a href="#cl-877">877</a>
<a href="#cl-878">878</a>
<a href="#cl-879">879</a>
<a href="#cl-880">880</a>
<a href="#cl-881">881</a>
<a href="#cl-882">882</a>
<a href="#cl-883">883</a>
<a href="#cl-884">884</a>
<a href="#cl-885">885</a>
<a href="#cl-886">886</a>
<a href="#cl-887">887</a>
<a href="#cl-888">888</a>
<a href="#cl-889">889</a>
<a href="#cl-890">890</a>
<a href="#cl-891">891</a>
<a href="#cl-892">892</a>
<a href="#cl-893">893</a>
<a href="#cl-894">894</a>
<a href="#cl-895">895</a>
<a href="#cl-896">896</a>
<a href="#cl-897">897</a>
<a href="#cl-898">898</a>
<a href="#cl-899">899</a>
<a href="#cl-900">900</a>
<a href="#cl-901">901</a>
<a href="#cl-902">902</a>
<a href="#cl-903">903</a>
<a href="#cl-904">904</a>
<a href="#cl-905">905</a>
<a href="#cl-906">906</a>
<a href="#cl-907">907</a>
<a href="#cl-908">908</a>
<a href="#cl-909">909</a>
<a href="#cl-910">910</a>
<a href="#cl-911">911</a>
<a href="#cl-912">912</a>
<a href="#cl-913">913</a>
<a href="#cl-914">914</a>
<a href="#cl-915">915</a>
<a href="#cl-916">916</a>
<a href="#cl-917">917</a>
<a href="#cl-918">918</a>
<a href="#cl-919">919</a>
<a href="#cl-920">920</a>
<a href="#cl-921">921</a>
<a href="#cl-922">922</a>
<a href="#cl-923">923</a>
<a href="#cl-924">924</a>
<a href="#cl-925">925</a>
<a href="#cl-926">926</a>
<a href="#cl-927">927</a>
<a href="#cl-928">928</a>
<a href="#cl-929">929</a>
<a href="#cl-930">930</a>
<a href="#cl-931">931</a>
<a href="#cl-932">932</a>
<a href="#cl-933">933</a>
<a href="#cl-934">934</a>
<a href="#cl-935">935</a>
<a href="#cl-936">936</a>
<a href="#cl-937">937</a>
<a href="#cl-938">938</a>
<a href="#cl-939">939</a>
<a href="#cl-940">940</a>
<a href="#cl-941">941</a>
<a href="#cl-942">942</a>
<a href="#cl-943">943</a>
<a href="#cl-944">944</a>
<a href="#cl-945">945</a>
<a href="#cl-946">946</a>
<a href="#cl-947">947</a>
<a href="#cl-948">948</a>
<a href="#cl-949">949</a>
<a href="#cl-950">950</a>
<a href="#cl-951">951</a>
<a href="#cl-952">952</a>
<a href="#cl-953">953</a>
<a href="#cl-954">954</a>
<a href="#cl-955">955</a>
<a href="#cl-956">956</a>
<a href="#cl-957">957</a>
<a href="#cl-958">958</a>
<a href="#cl-959">959</a>
<a href="#cl-960">960</a>
<a href="#cl-961">961</a>
<a href="#cl-962">962</a>
<a href="#cl-963">963</a>
<a href="#cl-964">964</a>
<a href="#cl-965">965</a>
<a href="#cl-966">966</a>
<a href="#cl-967">967</a>
<a href="#cl-968">968</a>
<a href="#cl-969">969</a>
<a href="#cl-970">970</a>
<a href="#cl-971">971</a>
<a href="#cl-972">972</a>
<a href="#cl-973">973</a>
<a href="#cl-974">974</a>
<a href="#cl-975">975</a>
<a href="#cl-976">976</a>
<a href="#cl-977">977</a>
<a href="#cl-978">978</a>
<a href="#cl-979">979</a>
<a href="#cl-980">980</a>
<a href="#cl-981">981</a>
<a href="#cl-982">982</a>
<a href="#cl-983">983</a>
<a href="#cl-984">984</a>
<a href="#cl-985">985</a>
<a href="#cl-986">986</a>
<a href="#cl-987">987</a>
<a href="#cl-988">988</a>
<a href="#cl-989">989</a>
<a href="#cl-990">990</a>
<a href="#cl-991">991</a>
<a href="#cl-992">992</a>
<a href="#cl-993">993</a>
<a href="#cl-994">994</a>
<a href="#cl-995">995</a>
<a href="#cl-996">996</a>
<a href="#cl-997">997</a>
<a href="#cl-998">998</a>
<a href="#cl-999">999</a>
<a href="#cl-1000">1000</a>
<a href="#cl-1001">1001</a>
<a href="#cl-1002">1002</a>
<a href="#cl-1003">1003</a>
<a href="#cl-1004">1004</a>
<a href="#cl-1005">1005</a>
<a href="#cl-1006">1006</a>
<a href="#cl-1007">1007</a>
<a href="#cl-1008">1008</a>
<a href="#cl-1009">1009</a>
<a href="#cl-1010">1010</a>
<a href="#cl-1011">1011</a>
<a href="#cl-1012">1012</a>
<a href="#cl-1013">1013</a>
<a href="#cl-1014">1014</a>
<a href="#cl-1015">1015</a>
<a href="#cl-1016">1016</a>
<a href="#cl-1017">1017</a>
<a href="#cl-1018">1018</a>
<a href="#cl-1019">1019</a>
<a href="#cl-1020">1020</a>
<a href="#cl-1021">1021</a>
<a href="#cl-1022">1022</a>
<a href="#cl-1023">1023</a>
<a href="#cl-1024">1024</a>
<a href="#cl-1025">1025</a>
<a href="#cl-1026">1026</a>
<a href="#cl-1027">1027</a>
<a href="#cl-1028">1028</a>
<a href="#cl-1029">1029</a>
<a href="#cl-1030">1030</a>
<a href="#cl-1031">1031</a>
<a href="#cl-1032">1032</a>
<a href="#cl-1033">1033</a>
<a href="#cl-1034">1034</a>
<a href="#cl-1035">1035</a>
<a href="#cl-1036">1036</a>
<a href="#cl-1037">1037</a>
<a href="#cl-1038">1038</a>
<a href="#cl-1039">1039</a>
<a href="#cl-1040">1040</a>
<a href="#cl-1041">1041</a>
<a href="#cl-1042">1042</a>
<a href="#cl-1043">1043</a>
<a href="#cl-1044">1044</a>
<a href="#cl-1045">1045</a>
<a href="#cl-1046">1046</a>
<a href="#cl-1047">1047</a>
<a href="#cl-1048">1048</a>
<a href="#cl-1049">1049</a>
<a href="#cl-1050">1050</a>
<a href="#cl-1051">1051</a>
<a href="#cl-1052">1052</a>
<a href="#cl-1053">1053</a>
<a href="#cl-1054">1054</a>
<a href="#cl-1055">1055</a>
<a href="#cl-1056">1056</a>
<a href="#cl-1057">1057</a>
<a href="#cl-1058">1058</a>
<a href="#cl-1059">1059</a>
<a href="#cl-1060">1060</a>
<a href="#cl-1061">1061</a>
<a href="#cl-1062">1062</a>
<a href="#cl-1063">1063</a>
<a href="#cl-1064">1064</a>
<a href="#cl-1065">1065</a>
<a href="#cl-1066">1066</a>
<a href="#cl-1067">1067</a>
<a href="#cl-1068">1068</a>
<a href="#cl-1069">1069</a>
<a href="#cl-1070">1070</a>
<a href="#cl-1071">1071</a>
<a href="#cl-1072">1072</a>
<a href="#cl-1073">1073</a>
<a href="#cl-1074">1074</a>
<a href="#cl-1075">1075</a>
<a href="#cl-1076">1076</a>
<a href="#cl-1077">1077</a>
<a href="#cl-1078">1078</a>
<a href="#cl-1079">1079</a>
<a href="#cl-1080">1080</a>
<a href="#cl-1081">1081</a>
<a href="#cl-1082">1082</a>
<a href="#cl-1083">1083</a>
<a href="#cl-1084">1084</a>
<a href="#cl-1085">1085</a>
<a href="#cl-1086">1086</a>
<a href="#cl-1087">1087</a>
<a href="#cl-1088">1088</a>
<a href="#cl-1089">1089</a>
<a href="#cl-1090">1090</a>
<a href="#cl-1091">1091</a>
<a href="#cl-1092">1092</a>
<a href="#cl-1093">1093</a>
<a href="#cl-1094">1094</a>
<a href="#cl-1095">1095</a>
<a href="#cl-1096">1096</a>
<a href="#cl-1097">1097</a>
<a href="#cl-1098">1098</a>
<a href="#cl-1099">1099</a>
<a href="#cl-1100">1100</a>
<a href="#cl-1101">1101</a>
<a href="#cl-1102">1102</a>
<a href="#cl-1103">1103</a>
<a href="#cl-1104">1104</a>
<a href="#cl-1105">1105</a>
<a href="#cl-1106">1106</a>
<a href="#cl-1107">1107</a>
<a href="#cl-1108">1108</a>
<a href="#cl-1109">1109</a>
<a href="#cl-1110">1110</a>
<a href="#cl-1111">1111</a>
<a href="#cl-1112">1112</a>
<a href="#cl-1113">1113</a>
<a href="#cl-1114">1114</a>
<a href="#cl-1115">1115</a>
<a href="#cl-1116">1116</a>
<a href="#cl-1117">1117</a>
<a href="#cl-1118">1118</a>
<a href="#cl-1119">1119</a>
<a href="#cl-1120">1120</a>
<a href="#cl-1121">1121</a>
<a href="#cl-1122">1122</a>
<a href="#cl-1123">1123</a>
<a href="#cl-1124">1124</a>
<a href="#cl-1125">1125</a>
<a href="#cl-1126">1126</a>
<a href="#cl-1127">1127</a>
<a href="#cl-1128">1128</a>
<a href="#cl-1129">1129</a>
<a href="#cl-1130">1130</a>
<a href="#cl-1131">1131</a>
<a href="#cl-1132">1132</a>
<a href="#cl-1133">1133</a>
<a href="#cl-1134">1134</a>
<a href="#cl-1135">1135</a>
<a href="#cl-1136">1136</a>
<a href="#cl-1137">1137</a>
<a href="#cl-1138">1138</a>
<a href="#cl-1139">1139</a>
<a href="#cl-1140">1140</a>
<a href="#cl-1141">1141</a>
<a href="#cl-1142">1142</a>
<a href="#cl-1143">1143</a>
<a href="#cl-1144">1144</a>
<a href="#cl-1145">1145</a>
<a href="#cl-1146">1146</a>
<a href="#cl-1147">1147</a>
<a href="#cl-1148">1148</a>
<a href="#cl-1149">1149</a>
<a href="#cl-1150">1150</a>
<a href="#cl-1151">1151</a>
<a href="#cl-1152">1152</a>
<a href="#cl-1153">1153</a>
<a href="#cl-1154">1154</a>
<a href="#cl-1155">1155</a>
<a href="#cl-1156">1156</a>
<a href="#cl-1157">1157</a>
<a href="#cl-1158">1158</a>
<a href="#cl-1159">1159</a>
<a href="#cl-1160">1160</a>
<a href="#cl-1161">1161</a>
<a href="#cl-1162">1162</a>
<a href="#cl-1163">1163</a>
<a href="#cl-1164">1164</a>
<a href="#cl-1165">1165</a>
<a href="#cl-1166">1166</a>
<a href="#cl-1167">1167</a>
<a href="#cl-1168">1168</a>
<a href="#cl-1169">1169</a>
<a href="#cl-1170">1170</a>
<a href="#cl-1171">1171</a>
<a href="#cl-1172">1172</a>
<a href="#cl-1173">1173</a>
<a href="#cl-1174">1174</a>
<a href="#cl-1175">1175</a>
<a href="#cl-1176">1176</a>
<a href="#cl-1177">1177</a>
<a href="#cl-1178">1178</a>
<a href="#cl-1179">1179</a>
<a href="#cl-1180">1180</a>
<a href="#cl-1181">1181</a>
<a href="#cl-1182">1182</a>
<a href="#cl-1183">1183</a>
<a href="#cl-1184">1184</a>
<a href="#cl-1185">1185</a>
<a href="#cl-1186">1186</a>
<a href="#cl-1187">1187</a>
<a href="#cl-1188">1188</a>
<a href="#cl-1189">1189</a>
<a href="#cl-1190">1190</a>
<a href="#cl-1191">1191</a>
<a href="#cl-1192">1192</a>
<a href="#cl-1193">1193</a>
<a href="#cl-1194">1194</a>
<a href="#cl-1195">1195</a>
<a href="#cl-1196">1196</a>
<a href="#cl-1197">1197</a>
<a href="#cl-1198">1198</a>
<a href="#cl-1199">1199</a>
<a href="#cl-1200">1200</a>
<a href="#cl-1201">1201</a>
<a href="#cl-1202">1202</a>
<a href="#cl-1203">1203</a>
<a href="#cl-1204">1204</a>
<a href="#cl-1205">1205</a>
<a href="#cl-1206">1206</a>
<a href="#cl-1207">1207</a>
<a href="#cl-1208">1208</a>
<a href="#cl-1209">1209</a>
<a href="#cl-1210">1210</a>
<a href="#cl-1211">1211</a>
<a href="#cl-1212">1212</a>
<a href="#cl-1213">1213</a>
<a href="#cl-1214">1214</a>
<a href="#cl-1215">1215</a>
<a href="#cl-1216">1216</a>
<a href="#cl-1217">1217</a>
<a href="#cl-1218">1218</a>
<a href="#cl-1219">1219</a>
<a href="#cl-1220">1220</a>
<a href="#cl-1221">1221</a>
<a href="#cl-1222">1222</a>
<a href="#cl-1223">1223</a>
<a href="#cl-1224">1224</a>
<a href="#cl-1225">1225</a>
<a href="#cl-1226">1226</a>
<a href="#cl-1227">1227</a>
<a href="#cl-1228">1228</a>
<a href="#cl-1229">1229</a>
<a href="#cl-1230">1230</a>
<a href="#cl-1231">1231</a>
<a href="#cl-1232">1232</a>
<a href="#cl-1233">1233</a>
<a href="#cl-1234">1234</a>
<a href="#cl-1235">1235</a>
<a href="#cl-1236">1236</a>
<a href="#cl-1237">1237</a>
<a href="#cl-1238">1238</a>
<a href="#cl-1239">1239</a>
<a href="#cl-1240">1240</a>
<a href="#cl-1241">1241</a>
<a href="#cl-1242">1242</a>
<a href="#cl-1243">1243</a>
<a href="#cl-1244">1244</a>
<a href="#cl-1245">1245</a>
<a href="#cl-1246">1246</a>
<a href="#cl-1247">1247</a>
<a href="#cl-1248">1248</a>
<a href="#cl-1249">1249</a>
<a href="#cl-1250">1250</a>
<a href="#cl-1251">1251</a>
<a href="#cl-1252">1252</a>
<a href="#cl-1253">1253</a>
<a href="#cl-1254">1254</a>
<a href="#cl-1255">1255</a>
<a href="#cl-1256">1256</a>
<a href="#cl-1257">1257</a>
<a href="#cl-1258">1258</a>
<a href="#cl-1259">1259</a>
<a href="#cl-1260">1260</a>
<a href="#cl-1261">1261</a>
<a href="#cl-1262">1262</a>
<a href="#cl-1263">1263</a>
<a href="#cl-1264">1264</a>
<a href="#cl-1265">1265</a>
<a href="#cl-1266">1266</a>
<a href="#cl-1267">1267</a>
<a href="#cl-1268">1268</a>
<a href="#cl-1269">1269</a>
<a href="#cl-1270">1270</a>
<a href="#cl-1271">1271</a>
<a href="#cl-1272">1272</a>
<a href="#cl-1273">1273</a>
<a href="#cl-1274">1274</a>
<a href="#cl-1275">1275</a>
<a href="#cl-1276">1276</a>
<a href="#cl-1277">1277</a>
<a href="#cl-1278">1278</a>
<a href="#cl-1279">1279</a>
<a href="#cl-1280">1280</a>
<a href="#cl-1281">1281</a>
<a href="#cl-1282">1282</a>
<a href="#cl-1283">1283</a>
<a href="#cl-1284">1284</a>
<a href="#cl-1285">1285</a>
<a href="#cl-1286">1286</a>
<a href="#cl-1287">1287</a>
<a href="#cl-1288">1288</a>
<a href="#cl-1289">1289</a>
<a href="#cl-1290">1290</a>
<a href="#cl-1291">1291</a>
<a href="#cl-1292">1292</a>
<a href="#cl-1293">1293</a>
<a href="#cl-1294">1294</a>
<a href="#cl-1295">1295</a>
<a href="#cl-1296">1296</a>
<a href="#cl-1297">1297</a>
<a href="#cl-1298">1298</a>
<a href="#cl-1299">1299</a>
<a href="#cl-1300">1300</a>
<a href="#cl-1301">1301</a>
<a href="#cl-1302">1302</a>
<a href="#cl-1303">1303</a>
<a href="#cl-1304">1304</a>
<a href="#cl-1305">1305</a>
<a href="#cl-1306">1306</a>
<a href="#cl-1307">1307</a>
<a href="#cl-1308">1308</a>
<a href="#cl-1309">1309</a>
<a href="#cl-1310">1310</a>
<a href="#cl-1311">1311</a>
<a href="#cl-1312">1312</a>
<a href="#cl-1313">1313</a>
<a href="#cl-1314">1314</a>
<a href="#cl-1315">1315</a>
<a href="#cl-1316">1316</a>
<a href="#cl-1317">1317</a>
<a href="#cl-1318">1318</a>
<a href="#cl-1319">1319</a>
<a href="#cl-1320">1320</a>
<a href="#cl-1321">1321</a>
<a href="#cl-1322">1322</a>
<a href="#cl-1323">1323</a>
<a href="#cl-1324">1324</a>
<a href="#cl-1325">1325</a>
<a href="#cl-1326">1326</a>
<a href="#cl-1327">1327</a>
<a href="#cl-1328">1328</a>
<a href="#cl-1329">1329</a>
<a href="#cl-1330">1330</a>
<a href="#cl-1331">1331</a>
<a href="#cl-1332">1332</a>
<a href="#cl-1333">1333</a>
<a href="#cl-1334">1334</a>
<a href="#cl-1335">1335</a>
<a href="#cl-1336">1336</a>
<a href="#cl-1337">1337</a>
<a href="#cl-1338">1338</a>
<a href="#cl-1339">1339</a>
<a href="#cl-1340">1340</a>
<a href="#cl-1341">1341</a>
<a href="#cl-1342">1342</a>
<a href="#cl-1343">1343</a>
<a href="#cl-1344">1344</a>
<a href="#cl-1345">1345</a>
<a href="#cl-1346">1346</a>
<a href="#cl-1347">1347</a>
<a href="#cl-1348">1348</a>
<a href="#cl-1349">1349</a>
<a href="#cl-1350">1350</a>
<a href="#cl-1351">1351</a>
<a href="#cl-1352">1352</a>
<a href="#cl-1353">1353</a>
<a href="#cl-1354">1354</a>
<a href="#cl-1355">1355</a>
<a href="#cl-1356">1356</a>
<a href="#cl-1357">1357</a>
<a href="#cl-1358">1358</a>
<a href="#cl-1359">1359</a>
<a href="#cl-1360">1360</a>
<a href="#cl-1361">1361</a>
<a href="#cl-1362">1362</a>
<a href="#cl-1363">1363</a>
<a href="#cl-1364">1364</a>
<a href="#cl-1365">1365</a>
<a href="#cl-1366">1366</a>
<a href="#cl-1367">1367</a>
<a href="#cl-1368">1368</a>
<a href="#cl-1369">1369</a>
<a href="#cl-1370">1370</a>
<a href="#cl-1371">1371</a>
<a href="#cl-1372">1372</a>
<a href="#cl-1373">1373</a>
<a href="#cl-1374">1374</a>
<a href="#cl-1375">1375</a>
<a href="#cl-1376">1376</a>
<a href="#cl-1377">1377</a>
<a href="#cl-1378">1378</a>
<a href="#cl-1379">1379</a>
<a href="#cl-1380">1380</a>
<a href="#cl-1381">1381</a>
<a href="#cl-1382">1382</a>
<a href="#cl-1383">1383</a>
<a href="#cl-1384">1384</a>
<a href="#cl-1385">1385</a>
<a href="#cl-1386">1386</a>
<a href="#cl-1387">1387</a>
<a href="#cl-1388">1388</a>
<a href="#cl-1389">1389</a>
<a href="#cl-1390">1390</a>
<a href="#cl-1391">1391</a>
<a href="#cl-1392">1392</a>
<a href="#cl-1393">1393</a>
<a href="#cl-1394">1394</a>
<a href="#cl-1395">1395</a>
<a href="#cl-1396">1396</a>
<a href="#cl-1397">1397</a>
<a href="#cl-1398">1398</a>
<a href="#cl-1399">1399</a>
<a href="#cl-1400">1400</a>
<a href="#cl-1401">1401</a>
<a href="#cl-1402">1402</a>
<a href="#cl-1403">1403</a>
<a href="#cl-1404">1404</a>
<a href="#cl-1405">1405</a>
<a href="#cl-1406">1406</a>
<a href="#cl-1407">1407</a>
<a href="#cl-1408">1408</a>
<a href="#cl-1409">1409</a>
<a href="#cl-1410">1410</a>
<a href="#cl-1411">1411</a>
<a href="#cl-1412">1412</a>
<a href="#cl-1413">1413</a>
<a href="#cl-1414">1414</a>
<a href="#cl-1415">1415</a>
<a href="#cl-1416">1416</a>
<a href="#cl-1417">1417</a>
<a href="#cl-1418">1418</a>
<a href="#cl-1419">1419</a>
<a href="#cl-1420">1420</a>
<a href="#cl-1421">1421</a>
<a href="#cl-1422">1422</a>
<a href="#cl-1423">1423</a>
<a href="#cl-1424">1424</a>
<a href="#cl-1425">1425</a>
<a href="#cl-1426">1426</a>
<a href="#cl-1427">1427</a>
<a href="#cl-1428">1428</a>
<a href="#cl-1429">1429</a>
<a href="#cl-1430">1430</a>
<a href="#cl-1431">1431</a>
<a href="#cl-1432">1432</a>
<a href="#cl-1433">1433</a>
<a href="#cl-1434">1434</a>
<a href="#cl-1435">1435</a>
<a href="#cl-1436">1436</a>
<a href="#cl-1437">1437</a>
<a href="#cl-1438">1438</a>
<a href="#cl-1439">1439</a>
<a href="#cl-1440">1440</a>
<a href="#cl-1441">1441</a>
<a href="#cl-1442">1442</a>
<a href="#cl-1443">1443</a>
<a href="#cl-1444">1444</a>
<a href="#cl-1445">1445</a>
<a href="#cl-1446">1446</a>
<a href="#cl-1447">1447</a>
<a href="#cl-1448">1448</a>
<a href="#cl-1449">1449</a>
<a href="#cl-1450">1450</a>
<a href="#cl-1451">1451</a>
<a href="#cl-1452">1452</a>
<a href="#cl-1453">1453</a>
<a href="#cl-1454">1454</a>
<a href="#cl-1455">1455</a>
<a href="#cl-1456">1456</a>
<a href="#cl-1457">1457</a>
<a href="#cl-1458">1458</a>
<a href="#cl-1459">1459</a>
<a href="#cl-1460">1460</a>
<a href="#cl-1461">1461</a>
<a href="#cl-1462">1462</a>
<a href="#cl-1463">1463</a>
<a href="#cl-1464">1464</a>
<a href="#cl-1465">1465</a>
<a href="#cl-1466">1466</a>
<a href="#cl-1467">1467</a>
<a href="#cl-1468">1468</a>
<a href="#cl-1469">1469</a>
<a href="#cl-1470">1470</a>
<a href="#cl-1471">1471</a>
<a href="#cl-1472">1472</a>
<a href="#cl-1473">1473</a>
<a href="#cl-1474">1474</a>
<a href="#cl-1475">1475</a>
<a href="#cl-1476">1476</a>
<a href="#cl-1477">1477</a>
<a href="#cl-1478">1478</a>
<a href="#cl-1479">1479</a>
<a href="#cl-1480">1480</a>
<a href="#cl-1481">1481</a>
<a href="#cl-1482">1482</a>
<a href="#cl-1483">1483</a>
<a href="#cl-1484">1484</a>
<a href="#cl-1485">1485</a>
<a href="#cl-1486">1486</a>
<a href="#cl-1487">1487</a>
<a href="#cl-1488">1488</a>
<a href="#cl-1489">1489</a>
<a href="#cl-1490">1490</a>
<a href="#cl-1491">1491</a>
<a href="#cl-1492">1492</a>
<a href="#cl-1493">1493</a>
<a href="#cl-1494">1494</a>
<a href="#cl-1495">1495</a>
<a href="#cl-1496">1496</a>
<a href="#cl-1497">1497</a>
<a href="#cl-1498">1498</a>
<a href="#cl-1499">1499</a>
<a href="#cl-1500">1500</a>
<a href="#cl-1501">1501</a>
<a href="#cl-1502">1502</a>
<a href="#cl-1503">1503</a>
<a href="#cl-1504">1504</a>
<a href="#cl-1505">1505</a>
<a href="#cl-1506">1506</a>
<a href="#cl-1507">1507</a>
<a href="#cl-1508">1508</a>
<a href="#cl-1509">1509</a>
<a href="#cl-1510">1510</a>
<a href="#cl-1511">1511</a>
<a href="#cl-1512">1512</a>
<a href="#cl-1513">1513</a>
<a href="#cl-1514">1514</a>
<a href="#cl-1515">1515</a>
<a href="#cl-1516">1516</a>
<a href="#cl-1517">1517</a>
<a href="#cl-1518">1518</a>
<a href="#cl-1519">1519</a>
<a href="#cl-1520">1520</a>
<a href="#cl-1521">1521</a>
<a href="#cl-1522">1522</a>
<a href="#cl-1523">1523</a>
<a href="#cl-1524">1524</a>
<a href="#cl-1525">1525</a>
<a href="#cl-1526">1526</a>
<a href="#cl-1527">1527</a>
<a href="#cl-1528">1528</a>
<a href="#cl-1529">1529</a>
<a href="#cl-1530">1530</a>
<a href="#cl-1531">1531</a>
<a href="#cl-1532">1532</a>
<a href="#cl-1533">1533</a>
<a href="#cl-1534">1534</a>
<a href="#cl-1535">1535</a>
<a href="#cl-1536">1536</a>
<a href="#cl-1537">1537</a>
<a href="#cl-1538">1538</a>
<a href="#cl-1539">1539</a>
<a href="#cl-1540">1540</a>
<a href="#cl-1541">1541</a>
<a href="#cl-1542">1542</a>
<a href="#cl-1543">1543</a>
<a href="#cl-1544">1544</a>
<a href="#cl-1545">1545</a>
<a href="#cl-1546">1546</a>
<a href="#cl-1547">1547</a>
<a href="#cl-1548">1548</a>
<a href="#cl-1549">1549</a>
<a href="#cl-1550">1550</a>
<a href="#cl-1551">1551</a>
<a href="#cl-1552">1552</a>
<a href="#cl-1553">1553</a>
<a href="#cl-1554">1554</a>
<a href="#cl-1555">1555</a>
<a href="#cl-1556">1556</a>
<a href="#cl-1557">1557</a>
<a href="#cl-1558">1558</a>
<a href="#cl-1559">1559</a>
<a href="#cl-1560">1560</a>
<a href="#cl-1561">1561</a>
<a href="#cl-1562">1562</a>
<a href="#cl-1563">1563</a>
<a href="#cl-1564">1564</a>
<a href="#cl-1565">1565</a>
<a href="#cl-1566">1566</a>
<a href="#cl-1567">1567</a>
<a href="#cl-1568">1568</a>
<a href="#cl-1569">1569</a>
<a href="#cl-1570">1570</a>
<a href="#cl-1571">1571</a>
<a href="#cl-1572">1572</a>
<a href="#cl-1573">1573</a>
<a href="#cl-1574">1574</a>
<a href="#cl-1575">1575</a>
<a href="#cl-1576">1576</a>
<a href="#cl-1577">1577</a>
<a href="#cl-1578">1578</a>
<a href="#cl-1579">1579</a>
<a href="#cl-1580">1580</a>
<a href="#cl-1581">1581</a>
<a href="#cl-1582">1582</a>
<a href="#cl-1583">1583</a>
<a href="#cl-1584">1584</a>
<a href="#cl-1585">1585</a>
<a href="#cl-1586">1586</a>
<a href="#cl-1587">1587</a>
<a href="#cl-1588">1588</a>
<a href="#cl-1589">1589</a>
<a href="#cl-1590">1590</a>
<a href="#cl-1591">1591</a>
<a href="#cl-1592">1592</a>
<a href="#cl-1593">1593</a>
<a href="#cl-1594">1594</a>
<a href="#cl-1595">1595</a>
<a href="#cl-1596">1596</a>
<a href="#cl-1597">1597</a>
<a href="#cl-1598">1598</a>
<a href="#cl-1599">1599</a>
<a href="#cl-1600">1600</a>
<a href="#cl-1601">1601</a>
<a href="#cl-1602">1602</a>
<a href="#cl-1603">1603</a>
<a href="#cl-1604">1604</a>
<a href="#cl-1605">1605</a>
<a href="#cl-1606">1606</a>
<a href="#cl-1607">1607</a>
<a href="#cl-1608">1608</a>
<a href="#cl-1609">1609</a>
<a href="#cl-1610">1610</a>
<a href="#cl-1611">1611</a>
<a href="#cl-1612">1612</a>
<a href="#cl-1613">1613</a>
<a href="#cl-1614">1614</a>
<a href="#cl-1615">1615</a>
<a href="#cl-1616">1616</a>
<a href="#cl-1617">1617</a>
<a href="#cl-1618">1618</a>
<a href="#cl-1619">1619</a>
<a href="#cl-1620">1620</a>
<a href="#cl-1621">1621</a>
<a href="#cl-1622">1622</a>
<a href="#cl-1623">1623</a>
<a href="#cl-1624">1624</a>
<a href="#cl-1625">1625</a>
<a href="#cl-1626">1626</a>
<a href="#cl-1627">1627</a>
<a href="#cl-1628">1628</a>
<a href="#cl-1629">1629</a>
<a href="#cl-1630">1630</a>
<a href="#cl-1631">1631</a>
<a href="#cl-1632">1632</a>
<a href="#cl-1633">1633</a>
<a href="#cl-1634">1634</a>
<a href="#cl-1635">1635</a>
<a href="#cl-1636">1636</a>
<a href="#cl-1637">1637</a>
<a href="#cl-1638">1638</a>
<a href="#cl-1639">1639</a>
<a href="#cl-1640">1640</a>
<a href="#cl-1641">1641</a>
<a href="#cl-1642">1642</a>
<a href="#cl-1643">1643</a>
<a href="#cl-1644">1644</a>
<a href="#cl-1645">1645</a>
<a href="#cl-1646">1646</a>
<a href="#cl-1647">1647</a>
<a href="#cl-1648">1648</a>
<a href="#cl-1649">1649</a>
<a href="#cl-1650">1650</a>
<a href="#cl-1651">1651</a>
<a href="#cl-1652">1652</a>
<a href="#cl-1653">1653</a>
<a href="#cl-1654">1654</a>
<a href="#cl-1655">1655</a>
<a href="#cl-1656">1656</a>
<a href="#cl-1657">1657</a>
<a href="#cl-1658">1658</a>
<a href="#cl-1659">1659</a>
<a href="#cl-1660">1660</a>
<a href="#cl-1661">1661</a>
<a href="#cl-1662">1662</a>
<a href="#cl-1663">1663</a>
<a href="#cl-1664">1664</a>
<a href="#cl-1665">1665</a>
<a href="#cl-1666">1666</a>
<a href="#cl-1667">1667</a>
<a href="#cl-1668">1668</a>
<a href="#cl-1669">1669</a>
<a href="#cl-1670">1670</a>
<a href="#cl-1671">1671</a>
<a href="#cl-1672">1672</a>
<a href="#cl-1673">1673</a>
<a href="#cl-1674">1674</a>
<a href="#cl-1675">1675</a>
<a href="#cl-1676">1676</a>
<a href="#cl-1677">1677</a>
<a href="#cl-1678">1678</a>
<a href="#cl-1679">1679</a>
<a href="#cl-1680">1680</a>
<a href="#cl-1681">1681</a>
<a href="#cl-1682">1682</a>
<a href="#cl-1683">1683</a>
<a href="#cl-1684">1684</a>
<a href="#cl-1685">1685</a>
<a href="#cl-1686">1686</a>
<a href="#cl-1687">1687</a>
<a href="#cl-1688">1688</a>
<a href="#cl-1689">1689</a>
<a href="#cl-1690">1690</a>
<a href="#cl-1691">1691</a>
<a href="#cl-1692">1692</a>
<a href="#cl-1693">1693</a>
<a href="#cl-1694">1694</a>
<a href="#cl-1695">1695</a>
<a href="#cl-1696">1696</a>
<a href="#cl-1697">1697</a>
<a href="#cl-1698">1698</a>
<a href="#cl-1699">1699</a>
<a href="#cl-1700">1700</a>
<a href="#cl-1701">1701</a>
<a href="#cl-1702">1702</a>
<a href="#cl-1703">1703</a>
<a href="#cl-1704">1704</a>
<a href="#cl-1705">1705</a>
<a href="#cl-1706">1706</a>
<a href="#cl-1707">1707</a>
<a href="#cl-1708">1708</a>
<a href="#cl-1709">1709</a>
<a href="#cl-1710">1710</a>
<a href="#cl-1711">1711</a>
<a href="#cl-1712">1712</a>
<a href="#cl-1713">1713</a>
<a href="#cl-1714">1714</a>
<a href="#cl-1715">1715</a>
<a href="#cl-1716">1716</a>
<a href="#cl-1717">1717</a>
<a href="#cl-1718">1718</a>
<a href="#cl-1719">1719</a>
<a href="#cl-1720">1720</a>
<a href="#cl-1721">1721</a>
<a href="#cl-1722">1722</a>
<a href="#cl-1723">1723</a>
<a href="#cl-1724">1724</a>
<a href="#cl-1725">1725</a>
<a href="#cl-1726">1726</a>
<a href="#cl-1727">1727</a>
<a href="#cl-1728">1728</a>
<a href="#cl-1729">1729</a>
<a href="#cl-1730">1730</a>
<a href="#cl-1731">1731</a>
<a href="#cl-1732">1732</a>
<a href="#cl-1733">1733</a>
<a href="#cl-1734">1734</a>
<a href="#cl-1735">1735</a>
<a href="#cl-1736">1736</a>
<a href="#cl-1737">1737</a>
<a href="#cl-1738">1738</a>
<a href="#cl-1739">1739</a>
<a href="#cl-1740">1740</a>
<a href="#cl-1741">1741</a>
<a href="#cl-1742">1742</a>
<a href="#cl-1743">1743</a>
<a href="#cl-1744">1744</a>
<a href="#cl-1745">1745</a>
<a href="#cl-1746">1746</a>
<a href="#cl-1747">1747</a>
<a href="#cl-1748">1748</a>
<a href="#cl-1749">1749</a>
<a href="#cl-1750">1750</a>
<a href="#cl-1751">1751</a>
<a href="#cl-1752">1752</a>
<a href="#cl-1753">1753</a>
<a href="#cl-1754">1754</a>
<a href="#cl-1755">1755</a>
<a href="#cl-1756">1756</a>
<a href="#cl-1757">1757</a>
<a href="#cl-1758">1758</a>
<a href="#cl-1759">1759</a>
<a href="#cl-1760">1760</a>
<a href="#cl-1761">1761</a>
<a href="#cl-1762">1762</a>
<a href="#cl-1763">1763</a>
<a href="#cl-1764">1764</a>
<a href="#cl-1765">1765</a>
<a href="#cl-1766">1766</a>
<a href="#cl-1767">1767</a>
<a href="#cl-1768">1768</a>
<a href="#cl-1769">1769</a>
<a href="#cl-1770">1770</a>
<a href="#cl-1771">1771</a>
<a href="#cl-1772">1772</a>
<a href="#cl-1773">1773</a>
<a href="#cl-1774">1774</a>
<a href="#cl-1775">1775</a>
<a href="#cl-1776">1776</a>
<a href="#cl-1777">1777</a>
<a href="#cl-1778">1778</a>
<a href="#cl-1779">1779</a>
<a href="#cl-1780">1780</a>
<a href="#cl-1781">1781</a>
<a href="#cl-1782">1782</a>
<a href="#cl-1783">1783</a>
<a href="#cl-1784">1784</a>
<a href="#cl-1785">1785</a>
<a href="#cl-1786">1786</a>
<a href="#cl-1787">1787</a>
<a href="#cl-1788">1788</a>
<a href="#cl-1789">1789</a>
<a href="#cl-1790">1790</a>
<a href="#cl-1791">1791</a>
<a href="#cl-1792">1792</a>
<a href="#cl-1793">1793</a>
<a href="#cl-1794">1794</a>
<a href="#cl-1795">1795</a>
<a href="#cl-1796">1796</a>
<a href="#cl-1797">1797</a>
<a href="#cl-1798">1798</a>
<a href="#cl-1799">1799</a>
<a href="#cl-1800">1800</a>
<a href="#cl-1801">1801</a>
<a href="#cl-1802">1802</a>
<a href="#cl-1803">1803</a>
<a href="#cl-1804">1804</a>
<a href="#cl-1805">1805</a>
<a href="#cl-1806">1806</a>
<a href="#cl-1807">1807</a>
<a href="#cl-1808">1808</a>
<a href="#cl-1809">1809</a>
<a href="#cl-1810">1810</a>
<a href="#cl-1811">1811</a>
<a href="#cl-1812">1812</a>
<a href="#cl-1813">1813</a>
<a href="#cl-1814">1814</a>
<a href="#cl-1815">1815</a>
<a href="#cl-1816">1816</a>
<a href="#cl-1817">1817</a>
<a href="#cl-1818">1818</a>
<a href="#cl-1819">1819</a>
<a href="#cl-1820">1820</a>
<a href="#cl-1821">1821</a>
<a href="#cl-1822">1822</a>
<a href="#cl-1823">1823</a>
<a href="#cl-1824">1824</a>
<a href="#cl-1825">1825</a>
<a href="#cl-1826">1826</a>
<a href="#cl-1827">1827</a>
<a href="#cl-1828">1828</a>
<a href="#cl-1829">1829</a>
<a href="#cl-1830">1830</a>
<a href="#cl-1831">1831</a>
<a href="#cl-1832">1832</a>
<a href="#cl-1833">1833</a>
<a href="#cl-1834">1834</a>
<a href="#cl-1835">1835</a>
<a href="#cl-1836">1836</a>
<a href="#cl-1837">1837</a>
<a href="#cl-1838">1838</a>
<a href="#cl-1839">1839</a>
<a href="#cl-1840">1840</a>
<a href="#cl-1841">1841</a>
<a href="#cl-1842">1842</a>
<a href="#cl-1843">1843</a>
<a href="#cl-1844">1844</a>
<a href="#cl-1845">1845</a>
<a href="#cl-1846">1846</a>
<a href="#cl-1847">1847</a>
<a href="#cl-1848">1848</a>
<a href="#cl-1849">1849</a>
<a href="#cl-1850">1850</a>
<a href="#cl-1851">1851</a>
<a href="#cl-1852">1852</a>
<a href="#cl-1853">1853</a>
<a href="#cl-1854">1854</a>
<a href="#cl-1855">1855</a>
<a href="#cl-1856">1856</a>
<a href="#cl-1857">1857</a>
<a href="#cl-1858">1858</a>
<a href="#cl-1859">1859</a>
<a href="#cl-1860">1860</a>
<a href="#cl-1861">1861</a>
<a href="#cl-1862">1862</a>
<a href="#cl-1863">1863</a>
<a href="#cl-1864">1864</a>
<a href="#cl-1865">1865</a>
<a href="#cl-1866">1866</a>
<a href="#cl-1867">1867</a>
<a href="#cl-1868">1868</a>
<a href="#cl-1869">1869</a>
<a href="#cl-1870">1870</a>
<a href="#cl-1871">1871</a>
<a href="#cl-1872">1872</a>
<a href="#cl-1873">1873</a>
<a href="#cl-1874">1874</a>
<a href="#cl-1875">1875</a>
<a href="#cl-1876">1876</a>
<a href="#cl-1877">1877</a>
<a href="#cl-1878">1878</a>
<a href="#cl-1879">1879</a>
<a href="#cl-1880">1880</a>
<a href="#cl-1881">1881</a>
<a href="#cl-1882">1882</a>
<a href="#cl-1883">1883</a>
<a href="#cl-1884">1884</a>
<a href="#cl-1885">1885</a>
<a href="#cl-1886">1886</a>
<a href="#cl-1887">1887</a>
<a href="#cl-1888">1888</a>
<a href="#cl-1889">1889</a>
<a href="#cl-1890">1890</a>
<a href="#cl-1891">1891</a>
<a href="#cl-1892">1892</a>
<a href="#cl-1893">1893</a>
<a href="#cl-1894">1894</a>
<a href="#cl-1895">1895</a>
<a href="#cl-1896">1896</a>
<a href="#cl-1897">1897</a>
<a href="#cl-1898">1898</a>
<a href="#cl-1899">1899</a>
<a href="#cl-1900">1900</a>
<a href="#cl-1901">1901</a>
<a href="#cl-1902">1902</a>
<a href="#cl-1903">1903</a>
<a href="#cl-1904">1904</a>
<a href="#cl-1905">1905</a>
<a href="#cl-1906">1906</a>
<a href="#cl-1907">1907</a>
<a href="#cl-1908">1908</a>
<a href="#cl-1909">1909</a>
<a href="#cl-1910">1910</a>
<a href="#cl-1911">1911</a>
<a href="#cl-1912">1912</a>
<a href="#cl-1913">1913</a>
<a href="#cl-1914">1914</a>
<a href="#cl-1915">1915</a>
<a href="#cl-1916">1916</a>
<a href="#cl-1917">1917</a>
<a href="#cl-1918">1918</a>
<a href="#cl-1919">1919</a>
<a href="#cl-1920">1920</a>
<a href="#cl-1921">1921</a>
<a href="#cl-1922">1922</a>
<a href="#cl-1923">1923</a>
<a href="#cl-1924">1924</a>
<a href="#cl-1925">1925</a>
<a href="#cl-1926">1926</a>
<a href="#cl-1927">1927</a>
<a href="#cl-1928">1928</a>
<a href="#cl-1929">1929</a>
<a href="#cl-1930">1930</a>
<a href="#cl-1931">1931</a>
<a href="#cl-1932">1932</a>
<a href="#cl-1933">1933</a>
<a href="#cl-1934">1934</a>
<a href="#cl-1935">1935</a>
<a href="#cl-1936">1936</a>
<a href="#cl-1937">1937</a>
<a href="#cl-1938">1938</a>
<a href="#cl-1939">1939</a>
<a href="#cl-1940">1940</a>
<a href="#cl-1941">1941</a>
<a href="#cl-1942">1942</a>
<a href="#cl-1943">1943</a>
<a href="#cl-1944">1944</a>
<a href="#cl-1945">1945</a>
<a href="#cl-1946">1946</a>
<a href="#cl-1947">1947</a>
<a href="#cl-1948">1948</a>
<a href="#cl-1949">1949</a>
<a href="#cl-1950">1950</a>
<a href="#cl-1951">1951</a>
<a href="#cl-1952">1952</a>
<a href="#cl-1953">1953</a>
<a href="#cl-1954">1954</a>
<a href="#cl-1955">1955</a>
<a href="#cl-1956">1956</a>
<a href="#cl-1957">1957</a>
<a href="#cl-1958">1958</a>
<a href="#cl-1959">1959</a>
<a href="#cl-1960">1960</a>
<a href="#cl-1961">1961</a>
<a href="#cl-1962">1962</a>
<a href="#cl-1963">1963</a>
<a href="#cl-1964">1964</a>
<a href="#cl-1965">1965</a>
<a href="#cl-1966">1966</a>
<a href="#cl-1967">1967</a>
<a href="#cl-1968">1968</a>
<a href="#cl-1969">1969</a>
<a href="#cl-1970">1970</a>
<a href="#cl-1971">1971</a>
<a href="#cl-1972">1972</a>
<a href="#cl-1973">1973</a>
<a href="#cl-1974">1974</a>
<a href="#cl-1975">1975</a>
<a href="#cl-1976">1976</a>
<a href="#cl-1977">1977</a>
<a href="#cl-1978">1978</a>
<a href="#cl-1979">1979</a>
<a href="#cl-1980">1980</a>
<a href="#cl-1981">1981</a>
<a href="#cl-1982">1982</a>
<a href="#cl-1983">1983</a>
<a href="#cl-1984">1984</a>
<a href="#cl-1985">1985</a>
<a href="#cl-1986">1986</a>
<a href="#cl-1987">1987</a>
<a href="#cl-1988">1988</a>
<a href="#cl-1989">1989</a>
<a href="#cl-1990">1990</a>
<a href="#cl-1991">1991</a>
<a href="#cl-1992">1992</a>
<a href="#cl-1993">1993</a>
<a href="#cl-1994">1994</a>
<a href="#cl-1995">1995</a>
<a href="#cl-1996">1996</a>
<a href="#cl-1997">1997</a>
<a href="#cl-1998">1998</a>
<a href="#cl-1999">1999</a>
<a href="#cl-2000">2000</a>
<a href="#cl-2001">2001</a>
<a href="#cl-2002">2002</a>
<a href="#cl-2003">2003</a>
<a href="#cl-2004">2004</a>
<a href="#cl-2005">2005</a>
<a href="#cl-2006">2006</a>
<a href="#cl-2007">2007</a>
<a href="#cl-2008">2008</a>
<a href="#cl-2009">2009</a>
<a href="#cl-2010">2010</a>
<a href="#cl-2011">2011</a>
<a href="#cl-2012">2012</a>
<a href="#cl-2013">2013</a>
<a href="#cl-2014">2014</a>
<a href="#cl-2015">2015</a>
<a href="#cl-2016">2016</a>
<a href="#cl-2017">2017</a>
<a href="#cl-2018">2018</a>
<a href="#cl-2019">2019</a>
<a href="#cl-2020">2020</a>
<a href="#cl-2021">2021</a>
<a href="#cl-2022">2022</a>
<a href="#cl-2023">2023</a>
<a href="#cl-2024">2024</a>
<a href="#cl-2025">2025</a>
<a href="#cl-2026">2026</a>
<a href="#cl-2027">2027</a>
<a href="#cl-2028">2028</a>
<a href="#cl-2029">2029</a>
<a href="#cl-2030">2030</a>
<a href="#cl-2031">2031</a>
<a href="#cl-2032">2032</a>
<a href="#cl-2033">2033</a>
<a href="#cl-2034">2034</a>
<a href="#cl-2035">2035</a>
<a href="#cl-2036">2036</a>
<a href="#cl-2037">2037</a>
<a href="#cl-2038">2038</a>
<a href="#cl-2039">2039</a>
<a href="#cl-2040">2040</a>
<a href="#cl-2041">2041</a>
<a href="#cl-2042">2042</a>
<a href="#cl-2043">2043</a>
<a href="#cl-2044">2044</a>
<a href="#cl-2045">2045</a>
<a href="#cl-2046">2046</a>
<a href="#cl-2047">2047</a>
<a href="#cl-2048">2048</a>
<a href="#cl-2049">2049</a>
<a href="#cl-2050">2050</a>
<a href="#cl-2051">2051</a>
<a href="#cl-2052">2052</a>
<a href="#cl-2053">2053</a>
<a href="#cl-2054">2054</a>
<a href="#cl-2055">2055</a>
<a href="#cl-2056">2056</a>
<a href="#cl-2057">2057</a>
<a href="#cl-2058">2058</a>
<a href="#cl-2059">2059</a>
<a href="#cl-2060">2060</a>
<a href="#cl-2061">2061</a>
<a href="#cl-2062">2062</a>
<a href="#cl-2063">2063</a>
<a href="#cl-2064">2064</a>
<a href="#cl-2065">2065</a>
<a href="#cl-2066">2066</a>
<a href="#cl-2067">2067</a>
<a href="#cl-2068">2068</a>
<a href="#cl-2069">2069</a>
<a href="#cl-2070">2070</a>
<a href="#cl-2071">2071</a>
<a href="#cl-2072">2072</a>
<a href="#cl-2073">2073</a>
<a href="#cl-2074">2074</a>
<a href="#cl-2075">2075</a>
<a href="#cl-2076">2076</a>
<a href="#cl-2077">2077</a>
<a href="#cl-2078">2078</a>
<a href="#cl-2079">2079</a>
<a href="#cl-2080">2080</a>
<a href="#cl-2081">2081</a>
<a href="#cl-2082">2082</a>
<a href="#cl-2083">2083</a>
<a href="#cl-2084">2084</a>
<a href="#cl-2085">2085</a>
<a href="#cl-2086">2086</a>
<a href="#cl-2087">2087</a>
<a href="#cl-2088">2088</a>
<a href="#cl-2089">2089</a>
<a href="#cl-2090">2090</a>
<a href="#cl-2091">2091</a>
<a href="#cl-2092">2092</a>
<a href="#cl-2093">2093</a>
<a href="#cl-2094">2094</a>
<a href="#cl-2095">2095</a>
<a href="#cl-2096">2096</a>
<a href="#cl-2097">2097</a>
<a href="#cl-2098">2098</a>
<a href="#cl-2099">2099</a>
<a href="#cl-2100">2100</a>
<a href="#cl-2101">2101</a>
<a href="#cl-2102">2102</a>
<a href="#cl-2103">2103</a>
<a href="#cl-2104">2104</a>
<a href="#cl-2105">2105</a>
<a href="#cl-2106">2106</a>
<a href="#cl-2107">2107</a>
<a href="#cl-2108">2108</a>
<a href="#cl-2109">2109</a>
<a href="#cl-2110">2110</a>
<a href="#cl-2111">2111</a>
<a href="#cl-2112">2112</a>
<a href="#cl-2113">2113</a>
<a href="#cl-2114">2114</a>
<a href="#cl-2115">2115</a>
<a href="#cl-2116">2116</a>
<a href="#cl-2117">2117</a>
<a href="#cl-2118">2118</a>
<a href="#cl-2119">2119</a>
<a href="#cl-2120">2120</a>
<a href="#cl-2121">2121</a>
<a href="#cl-2122">2122</a>
<a href="#cl-2123">2123</a>
<a href="#cl-2124">2124</a>
<a href="#cl-2125">2125</a>
<a href="#cl-2126">2126</a>
<a href="#cl-2127">2127</a>
<a href="#cl-2128">2128</a>
<a href="#cl-2129">2129</a>
<a href="#cl-2130">2130</a>
<a href="#cl-2131">2131</a>
<a href="#cl-2132">2132</a>
<a href="#cl-2133">2133</a>
<a href="#cl-2134">2134</a>
<a href="#cl-2135">2135</a>
<a href="#cl-2136">2136</a>
<a href="#cl-2137">2137</a>
<a href="#cl-2138">2138</a>
<a href="#cl-2139">2139</a>
<a href="#cl-2140">2140</a>
<a href="#cl-2141">2141</a>
<a href="#cl-2142">2142</a>
<a href="#cl-2143">2143</a>
<a href="#cl-2144">2144</a>
<a href="#cl-2145">2145</a>
<a href="#cl-2146">2146</a>
<a href="#cl-2147">2147</a>
<a href="#cl-2148">2148</a>
<a href="#cl-2149">2149</a>
<a href="#cl-2150">2150</a>
<a href="#cl-2151">2151</a>
<a href="#cl-2152">2152</a>
<a href="#cl-2153">2153</a>
<a href="#cl-2154">2154</a>
<a href="#cl-2155">2155</a>
<a href="#cl-2156">2156</a>
<a href="#cl-2157">2157</a>
<a href="#cl-2158">2158</a>
<a href="#cl-2159">2159</a>
<a href="#cl-2160">2160</a>
<a href="#cl-2161">2161</a>
<a href="#cl-2162">2162</a>
<a href="#cl-2163">2163</a>
<a href="#cl-2164">2164</a>
<a href="#cl-2165">2165</a>
<a href="#cl-2166">2166</a>
<a href="#cl-2167">2167</a>
<a href="#cl-2168">2168</a>
<a href="#cl-2169">2169</a>
<a href="#cl-2170">2170</a>
<a href="#cl-2171">2171</a>
<a href="#cl-2172">2172</a>
<a href="#cl-2173">2173</a>
<a href="#cl-2174">2174</a>
<a href="#cl-2175">2175</a>
<a href="#cl-2176">2176</a>
<a href="#cl-2177">2177</a>
<a href="#cl-2178">2178</a>
<a href="#cl-2179">2179</a>
<a href="#cl-2180">2180</a>
<a href="#cl-2181">2181</a>
<a href="#cl-2182">2182</a>
<a href="#cl-2183">2183</a>
<a href="#cl-2184">2184</a>
<a href="#cl-2185">2185</a>
<a href="#cl-2186">2186</a>
<a href="#cl-2187">2187</a>
<a href="#cl-2188">2188</a>
<a href="#cl-2189">2189</a>
<a href="#cl-2190">2190</a>
<a href="#cl-2191">2191</a>
<a href="#cl-2192">2192</a>
<a href="#cl-2193">2193</a>
<a href="#cl-2194">2194</a>
<a href="#cl-2195">2195</a>
<a href="#cl-2196">2196</a>
<a href="#cl-2197">2197</a>
<a href="#cl-2198">2198</a>
<a href="#cl-2199">2199</a>
<a href="#cl-2200">2200</a>
<a href="#cl-2201">2201</a>
<a href="#cl-2202">2202</a>
<a href="#cl-2203">2203</a>
<a href="#cl-2204">2204</a>
<a href="#cl-2205">2205</a>
<a href="#cl-2206">2206</a>
<a href="#cl-2207">2207</a>
<a href="#cl-2208">2208</a>
<a href="#cl-2209">2209</a>
<a href="#cl-2210">2210</a>
<a href="#cl-2211">2211</a>
<a href="#cl-2212">2212</a>
<a href="#cl-2213">2213</a>
<a href="#cl-2214">2214</a>
<a href="#cl-2215">2215</a>
<a href="#cl-2216">2216</a>
<a href="#cl-2217">2217</a>
<a href="#cl-2218">2218</a>
<a href="#cl-2219">2219</a>
<a href="#cl-2220">2220</a>
<a href="#cl-2221">2221</a>
<a href="#cl-2222">2222</a>
<a href="#cl-2223">2223</a>
<a href="#cl-2224">2224</a>
<a href="#cl-2225">2225</a>
<a href="#cl-2226">2226</a>
<a href="#cl-2227">2227</a>
<a href="#cl-2228">2228</a>
<a href="#cl-2229">2229</a>
<a href="#cl-2230">2230</a>
<a href="#cl-2231">2231</a>
<a href="#cl-2232">2232</a>
<a href="#cl-2233">2233</a>
<a href="#cl-2234">2234</a>
<a href="#cl-2235">2235</a>
<a href="#cl-2236">2236</a>
<a href="#cl-2237">2237</a>
<a href="#cl-2238">2238</a>
<a href="#cl-2239">2239</a>
<a href="#cl-2240">2240</a>
<a href="#cl-2241">2241</a>
<a href="#cl-2242">2242</a>
<a href="#cl-2243">2243</a>
<a href="#cl-2244">2244</a>
<a href="#cl-2245">2245</a>
<a href="#cl-2246">2246</a>
<a href="#cl-2247">2247</a>
<a href="#cl-2248">2248</a>
<a href="#cl-2249">2249</a>
<a href="#cl-2250">2250</a>
<a href="#cl-2251">2251</a>
<a href="#cl-2252">2252</a>
<a href="#cl-2253">2253</a>
<a href="#cl-2254">2254</a>
<a href="#cl-2255">2255</a>
<a href="#cl-2256">2256</a>
<a href="#cl-2257">2257</a>
<a href="#cl-2258">2258</a>
<a href="#cl-2259">2259</a>
<a href="#cl-2260">2260</a>
<a href="#cl-2261">2261</a>
<a href="#cl-2262">2262</a>
<a href="#cl-2263">2263</a>
<a href="#cl-2264">2264</a>
<a href="#cl-2265">2265</a>
<a href="#cl-2266">2266</a>
<a href="#cl-2267">2267</a>
<a href="#cl-2268">2268</a>
<a href="#cl-2269">2269</a>
<a href="#cl-2270">2270</a>
<a href="#cl-2271">2271</a>
<a href="#cl-2272">2272</a>
<a href="#cl-2273">2273</a>
<a href="#cl-2274">2274</a>
<a href="#cl-2275">2275</a>
<a href="#cl-2276">2276</a>
<a href="#cl-2277">2277</a>
<a href="#cl-2278">2278</a>
<a href="#cl-2279">2279</a>
<a href="#cl-2280">2280</a>
<a href="#cl-2281">2281</a>
<a href="#cl-2282">2282</a>
<a href="#cl-2283">2283</a>
<a href="#cl-2284">2284</a>
<a href="#cl-2285">2285</a>
<a href="#cl-2286">2286</a>
<a href="#cl-2287">2287</a>
<a href="#cl-2288">2288</a>
<a href="#cl-2289">2289</a>
<a href="#cl-2290">2290</a>
<a href="#cl-2291">2291</a>
<a href="#cl-2292">2292</a>
<a href="#cl-2293">2293</a>
<a href="#cl-2294">2294</a>
<a href="#cl-2295">2295</a>
<a href="#cl-2296">2296</a>
<a href="#cl-2297">2297</a>
<a href="#cl-2298">2298</a>
<a href="#cl-2299">2299</a>
<a href="#cl-2300">2300</a>
<a href="#cl-2301">2301</a>
<a href="#cl-2302">2302</a>
<a href="#cl-2303">2303</a>
<a href="#cl-2304">2304</a>
<a href="#cl-2305">2305</a>
<a href="#cl-2306">2306</a>
<a href="#cl-2307">2307</a>
<a href="#cl-2308">2308</a>
<a href="#cl-2309">2309</a>
<a href="#cl-2310">2310</a>
<a href="#cl-2311">2311</a>
<a href="#cl-2312">2312</a>
<a href="#cl-2313">2313</a>
<a href="#cl-2314">2314</a>
<a href="#cl-2315">2315</a>
<a href="#cl-2316">2316</a>
<a href="#cl-2317">2317</a>
<a href="#cl-2318">2318</a>
<a href="#cl-2319">2319</a>
<a href="#cl-2320">2320</a>
<a href="#cl-2321">2321</a>
<a href="#cl-2322">2322</a>
<a href="#cl-2323">2323</a>
<a href="#cl-2324">2324</a>
<a href="#cl-2325">2325</a>
<a href="#cl-2326">2326</a>
<a href="#cl-2327">2327</a>
<a href="#cl-2328">2328</a>
<a href="#cl-2329">2329</a>
<a href="#cl-2330">2330</a>
<a href="#cl-2331">2331</a>
<a href="#cl-2332">2332</a>
<a href="#cl-2333">2333</a>
<a href="#cl-2334">2334</a>
<a href="#cl-2335">2335</a>
<a href="#cl-2336">2336</a>
<a href="#cl-2337">2337</a>
<a href="#cl-2338">2338</a>
<a href="#cl-2339">2339</a>
<a href="#cl-2340">2340</a>
<a href="#cl-2341">2341</a>
<a href="#cl-2342">2342</a>
<a href="#cl-2343">2343</a>
<a href="#cl-2344">2344</a>
<a href="#cl-2345">2345</a>
<a href="#cl-2346">2346</a>
<a href="#cl-2347">2347</a>
<a href="#cl-2348">2348</a>
<a href="#cl-2349">2349</a>
<a href="#cl-2350">2350</a>
<a href="#cl-2351">2351</a>
<a href="#cl-2352">2352</a>
<a href="#cl-2353">2353</a>
<a href="#cl-2354">2354</a>
<a href="#cl-2355">2355</a>
<a href="#cl-2356">2356</a>
<a href="#cl-2357">2357</a>
<a href="#cl-2358">2358</a>
<a href="#cl-2359">2359</a>
<a href="#cl-2360">2360</a>
<a href="#cl-2361">2361</a>
<a href="#cl-2362">2362</a>
<a href="#cl-2363">2363</a>
<a href="#cl-2364">2364</a>
<a href="#cl-2365">2365</a>
<a href="#cl-2366">2366</a>
<a href="#cl-2367">2367</a>
<a href="#cl-2368">2368</a>
<a href="#cl-2369">2369</a>
<a href="#cl-2370">2370</a>
<a href="#cl-2371">2371</a>
<a href="#cl-2372">2372</a>
<a href="#cl-2373">2373</a>
<a href="#cl-2374">2374</a>
<a href="#cl-2375">2375</a>
<a href="#cl-2376">2376</a>
<a href="#cl-2377">2377</a>
<a href="#cl-2378">2378</a>
<a href="#cl-2379">2379</a>
<a href="#cl-2380">2380</a>
<a href="#cl-2381">2381</a>
<a href="#cl-2382">2382</a>
<a href="#cl-2383">2383</a>
<a href="#cl-2384">2384</a>
<a href="#cl-2385">2385</a>
<a href="#cl-2386">2386</a>
<a href="#cl-2387">2387</a>
<a href="#cl-2388">2388</a>
<a href="#cl-2389">2389</a>
<a href="#cl-2390">2390</a>
<a href="#cl-2391">2391</a>
<a href="#cl-2392">2392</a>
<a href="#cl-2393">2393</a>
<a href="#cl-2394">2394</a>
<a href="#cl-2395">2395</a>
<a href="#cl-2396">2396</a>
<a href="#cl-2397">2397</a>
<a href="#cl-2398">2398</a>
<a href="#cl-2399">2399</a>
<a href="#cl-2400">2400</a>
<a href="#cl-2401">2401</a>
<a href="#cl-2402">2402</a>
<a href="#cl-2403">2403</a>
<a href="#cl-2404">2404</a>
<a href="#cl-2405">2405</a>
<a href="#cl-2406">2406</a>
<a href="#cl-2407">2407</a>
<a href="#cl-2408">2408</a>
<a href="#cl-2409">2409</a>
<a href="#cl-2410">2410</a>
<a href="#cl-2411">2411</a>
<a href="#cl-2412">2412</a>
<a href="#cl-2413">2413</a>
<a href="#cl-2414">2414</a>
<a href="#cl-2415">2415</a>
<a href="#cl-2416">2416</a>
<a href="#cl-2417">2417</a>
<a href="#cl-2418">2418</a>
<a href="#cl-2419">2419</a>
<a href="#cl-2420">2420</a>
<a href="#cl-2421">2421</a>
<a href="#cl-2422">2422</a>
<a href="#cl-2423">2423</a>
<a href="#cl-2424">2424</a>
<a href="#cl-2425">2425</a>
<a href="#cl-2426">2426</a>
<a href="#cl-2427">2427</a>
<a href="#cl-2428">2428</a>
<a href="#cl-2429">2429</a>
<a href="#cl-2430">2430</a>
<a href="#cl-2431">2431</a>
<a href="#cl-2432">2432</a>
<a href="#cl-2433">2433</a>
<a href="#cl-2434">2434</a>
<a href="#cl-2435">2435</a>
<a href="#cl-2436">2436</a>
<a href="#cl-2437">2437</a>
<a href="#cl-2438">2438</a>
<a href="#cl-2439">2439</a>
<a href="#cl-2440">2440</a>
<a href="#cl-2441">2441</a>
<a href="#cl-2442">2442</a>
<a href="#cl-2443">2443</a>
<a href="#cl-2444">2444</a>
<a href="#cl-2445">2445</a>
<a href="#cl-2446">2446</a>
<a href="#cl-2447">2447</a>
<a href="#cl-2448">2448</a>
<a href="#cl-2449">2449</a>
<a href="#cl-2450">2450</a>
<a href="#cl-2451">2451</a>
<a href="#cl-2452">2452</a>
<a href="#cl-2453">2453</a>
<a href="#cl-2454">2454</a>
<a href="#cl-2455">2455</a>
<a href="#cl-2456">2456</a>
<a href="#cl-2457">2457</a>
<a href="#cl-2458">2458</a>
<a href="#cl-2459">2459</a>
<a href="#cl-2460">2460</a>
<a href="#cl-2461">2461</a>
<a href="#cl-2462">2462</a>
<a href="#cl-2463">2463</a>
<a href="#cl-2464">2464</a>
<a href="#cl-2465">2465</a>
<a href="#cl-2466">2466</a>
<a href="#cl-2467">2467</a>
<a href="#cl-2468">2468</a>
<a href="#cl-2469">2469</a>
<a href="#cl-2470">2470</a>
<a href="#cl-2471">2471</a>
<a href="#cl-2472">2472</a>
<a href="#cl-2473">2473</a>
<a href="#cl-2474">2474</a>
<a href="#cl-2475">2475</a>
<a href="#cl-2476">2476</a>
<a href="#cl-2477">2477</a>
<a href="#cl-2478">2478</a>
<a href="#cl-2479">2479</a>
<a href="#cl-2480">2480</a>
<a href="#cl-2481">2481</a>
<a href="#cl-2482">2482</a>
<a href="#cl-2483">2483</a>
<a href="#cl-2484">2484</a>
<a href="#cl-2485">2485</a>
<a href="#cl-2486">2486</a>
<a href="#cl-2487">2487</a>
<a href="#cl-2488">2488</a>
<a href="#cl-2489">2489</a>
<a href="#cl-2490">2490</a>
<a href="#cl-2491">2491</a>
<a href="#cl-2492">2492</a>
<a href="#cl-2493">2493</a>
<a href="#cl-2494">2494</a>
<a href="#cl-2495">2495</a>
<a href="#cl-2496">2496</a>
<a href="#cl-2497">2497</a>
<a href="#cl-2498">2498</a>
<a href="#cl-2499">2499</a>
<a href="#cl-2500">2500</a>
<a href="#cl-2501">2501</a>
<a href="#cl-2502">2502</a>
<a href="#cl-2503">2503</a>
<a href="#cl-2504">2504</a>
<a href="#cl-2505">2505</a>
<a href="#cl-2506">2506</a>
<a href="#cl-2507">2507</a>
<a href="#cl-2508">2508</a>
<a href="#cl-2509">2509</a>
<a href="#cl-2510">2510</a>
<a href="#cl-2511">2511</a>
<a href="#cl-2512">2512</a>
<a href="#cl-2513">2513</a>
<a href="#cl-2514">2514</a>
<a href="#cl-2515">2515</a>
<a href="#cl-2516">2516</a>
<a href="#cl-2517">2517</a>
<a href="#cl-2518">2518</a>
<a href="#cl-2519">2519</a>
<a href="#cl-2520">2520</a>
<a href="#cl-2521">2521</a>
<a href="#cl-2522">2522</a>
<a href="#cl-2523">2523</a>
<a href="#cl-2524">2524</a>
<a href="#cl-2525">2525</a>
<a href="#cl-2526">2526</a>
<a href="#cl-2527">2527</a>
<a href="#cl-2528">2528</a>
<a href="#cl-2529">2529</a>
<a href="#cl-2530">2530</a>
<a href="#cl-2531">2531</a>
<a href="#cl-2532">2532</a>
<a href="#cl-2533">2533</a>
<a href="#cl-2534">2534</a>
<a href="#cl-2535">2535</a>
<a href="#cl-2536">2536</a>
<a href="#cl-2537">2537</a>
<a href="#cl-2538">2538</a>
<a href="#cl-2539">2539</a>
<a href="#cl-2540">2540</a>
<a href="#cl-2541">2541</a>
<a href="#cl-2542">2542</a>
<a href="#cl-2543">2543</a>
<a href="#cl-2544">2544</a>
<a href="#cl-2545">2545</a>
<a href="#cl-2546">2546</a>
<a href="#cl-2547">2547</a>
<a href="#cl-2548">2548</a>
<a href="#cl-2549">2549</a>
<a href="#cl-2550">2550</a>
<a href="#cl-2551">2551</a>
<a href="#cl-2552">2552</a>
<a href="#cl-2553">2553</a>
<a href="#cl-2554">2554</a>
<a href="#cl-2555">2555</a>
<a href="#cl-2556">2556</a>
<a href="#cl-2557">2557</a>
<a href="#cl-2558">2558</a>
<a href="#cl-2559">2559</a>
<a href="#cl-2560">2560</a>
<a href="#cl-2561">2561</a>
<a href="#cl-2562">2562</a>
<a href="#cl-2563">2563</a>
<a href="#cl-2564">2564</a>
<a href="#cl-2565">2565</a>
<a href="#cl-2566">2566</a>
<a href="#cl-2567">2567</a>
<a href="#cl-2568">2568</a>
<a href="#cl-2569">2569</a>
<a href="#cl-2570">2570</a>
<a href="#cl-2571">2571</a>
<a href="#cl-2572">2572</a>
<a href="#cl-2573">2573</a>
<a href="#cl-2574">2574</a>
<a href="#cl-2575">2575</a>
<a href="#cl-2576">2576</a>
<a href="#cl-2577">2577</a>
<a href="#cl-2578">2578</a>
<a href="#cl-2579">2579</a>
<a href="#cl-2580">2580</a>
<a href="#cl-2581">2581</a>
<a href="#cl-2582">2582</a>
<a href="#cl-2583">2583</a>
<a href="#cl-2584">2584</a>
<a href="#cl-2585">2585</a>
<a href="#cl-2586">2586</a>
<a href="#cl-2587">2587</a>
<a href="#cl-2588">2588</a>
<a href="#cl-2589">2589</a>
<a href="#cl-2590">2590</a>
<a href="#cl-2591">2591</a>
<a href="#cl-2592">2592</a>
<a href="#cl-2593">2593</a>
<a href="#cl-2594">2594</a>
<a href="#cl-2595">2595</a>
<a href="#cl-2596">2596</a>
<a href="#cl-2597">2597</a>
<a href="#cl-2598">2598</a>
<a href="#cl-2599">2599</a>
<a href="#cl-2600">2600</a>
<a href="#cl-2601">2601</a>
<a href="#cl-2602">2602</a>
<a href="#cl-2603">2603</a>
<a href="#cl-2604">2604</a>
<a href="#cl-2605">2605</a>
<a href="#cl-2606">2606</a>
<a href="#cl-2607">2607</a>
<a href="#cl-2608">2608</a>
<a href="#cl-2609">2609</a>
<a href="#cl-2610">2610</a>
<a href="#cl-2611">2611</a>
<a href="#cl-2612">2612</a>
<a href="#cl-2613">2613</a>
<a href="#cl-2614">2614</a>
<a href="#cl-2615">2615</a>
<a href="#cl-2616">2616</a>
<a href="#cl-2617">2617</a>
<a href="#cl-2618">2618</a>
<a href="#cl-2619">2619</a>
<a href="#cl-2620">2620</a>
<a href="#cl-2621">2621</a>
<a href="#cl-2622">2622</a>
<a href="#cl-2623">2623</a>
<a href="#cl-2624">2624</a>
<a href="#cl-2625">2625</a>
<a href="#cl-2626">2626</a>
<a href="#cl-2627">2627</a>
<a href="#cl-2628">2628</a>
<a href="#cl-2629">2629</a>
<a href="#cl-2630">2630</a>
<a href="#cl-2631">2631</a>
<a href="#cl-2632">2632</a>
<a href="#cl-2633">2633</a>
<a href="#cl-2634">2634</a>
<a href="#cl-2635">2635</a>
<a href="#cl-2636">2636</a>
<a href="#cl-2637">2637</a>
<a href="#cl-2638">2638</a>
<a href="#cl-2639">2639</a>
<a href="#cl-2640">2640</a>
<a href="#cl-2641">2641</a>
<a href="#cl-2642">2642</a>
<a href="#cl-2643">2643</a>
<a href="#cl-2644">2644</a>
<a href="#cl-2645">2645</a>
<a href="#cl-2646">2646</a>
<a href="#cl-2647">2647</a>
<a href="#cl-2648">2648</a>
<a href="#cl-2649">2649</a>
<a href="#cl-2650">2650</a>
<a href="#cl-2651">2651</a>
<a href="#cl-2652">2652</a>
<a href="#cl-2653">2653</a>
<a href="#cl-2654">2654</a>
<a href="#cl-2655">2655</a>
<a href="#cl-2656">2656</a>
<a href="#cl-2657">2657</a>
<a href="#cl-2658">2658</a>
<a href="#cl-2659">2659</a>
<a href="#cl-2660">2660</a>
<a href="#cl-2661">2661</a>
<a href="#cl-2662">2662</a>
<a href="#cl-2663">2663</a>
<a href="#cl-2664">2664</a>
<a href="#cl-2665">2665</a>
<a href="#cl-2666">2666</a>
<a href="#cl-2667">2667</a>
<a href="#cl-2668">2668</a>
<a href="#cl-2669">2669</a>
<a href="#cl-2670">2670</a>
<a href="#cl-2671">2671</a>
<a href="#cl-2672">2672</a>
<a href="#cl-2673">2673</a>
<a href="#cl-2674">2674</a>
<a href="#cl-2675">2675</a>
<a href="#cl-2676">2676</a>
<a href="#cl-2677">2677</a>
<a href="#cl-2678">2678</a>
<a href="#cl-2679">2679</a>
<a href="#cl-2680">2680</a>
<a href="#cl-2681">2681</a>
<a href="#cl-2682">2682</a>
<a href="#cl-2683">2683</a>
<a href="#cl-2684">2684</a>
<a href="#cl-2685">2685</a>
<a href="#cl-2686">2686</a>
<a href="#cl-2687">2687</a>
<a href="#cl-2688">2688</a>
<a href="#cl-2689">2689</a>
<a href="#cl-2690">2690</a>
<a href="#cl-2691">2691</a>
<a href="#cl-2692">2692</a>
<a href="#cl-2693">2693</a>
<a href="#cl-2694">2694</a>
<a href="#cl-2695">2695</a>
<a href="#cl-2696">2696</a>
<a href="#cl-2697">2697</a>
<a href="#cl-2698">2698</a>
<a href="#cl-2699">2699</a>
<a href="#cl-2700">2700</a>
<a href="#cl-2701">2701</a>
<a href="#cl-2702">2702</a>
<a href="#cl-2703">2703</a>
<a href="#cl-2704">2704</a>
<a href="#cl-2705">2705</a>
<a href="#cl-2706">2706</a>
<a href="#cl-2707">2707</a>
<a href="#cl-2708">2708</a>
<a href="#cl-2709">2709</a>
<a href="#cl-2710">2710</a>
<a href="#cl-2711">2711</a>
<a href="#cl-2712">2712</a>
<a href="#cl-2713">2713</a>
<a href="#cl-2714">2714</a>
<a href="#cl-2715">2715</a>
<a href="#cl-2716">2716</a>
<a href="#cl-2717">2717</a>
<a href="#cl-2718">2718</a>
<a href="#cl-2719">2719</a>
<a href="#cl-2720">2720</a>
<a href="#cl-2721">2721</a>
<a href="#cl-2722">2722</a>
<a href="#cl-2723">2723</a>
<a href="#cl-2724">2724</a>
<a href="#cl-2725">2725</a>
<a href="#cl-2726">2726</a>
<a href="#cl-2727">2727</a>
<a href="#cl-2728">2728</a>
<a href="#cl-2729">2729</a>
<a href="#cl-2730">2730</a>
<a href="#cl-2731">2731</a>
<a href="#cl-2732">2732</a>
<a href="#cl-2733">2733</a>
<a href="#cl-2734">2734</a>
<a href="#cl-2735">2735</a>
<a href="#cl-2736">2736</a>
<a href="#cl-2737">2737</a>
<a href="#cl-2738">2738</a>
<a href="#cl-2739">2739</a>
<a href="#cl-2740">2740</a>
<a href="#cl-2741">2741</a>
<a href="#cl-2742">2742</a>
<a href="#cl-2743">2743</a>
<a href="#cl-2744">2744</a>
<a href="#cl-2745">2745</a>
<a href="#cl-2746">2746</a>
<a href="#cl-2747">2747</a>
<a href="#cl-2748">2748</a>
<a href="#cl-2749">2749</a>
<a href="#cl-2750">2750</a>
<a href="#cl-2751">2751</a>
<a href="#cl-2752">2752</a>
<a href="#cl-2753">2753</a>
<a href="#cl-2754">2754</a>
<a href="#cl-2755">2755</a>
<a href="#cl-2756">2756</a>
<a href="#cl-2757">2757</a>
<a href="#cl-2758">2758</a>
<a href="#cl-2759">2759</a>
<a href="#cl-2760">2760</a>
<a href="#cl-2761">2761</a>
<a href="#cl-2762">2762</a>
<a href="#cl-2763">2763</a>
<a href="#cl-2764">2764</a>
<a href="#cl-2765">2765</a>
<a href="#cl-2766">2766</a>
<a href="#cl-2767">2767</a>
<a href="#cl-2768">2768</a>
<a href="#cl-2769">2769</a>
<a href="#cl-2770">2770</a>
<a href="#cl-2771">2771</a>
<a href="#cl-2772">2772</a>
<a href="#cl-2773">2773</a>
<a href="#cl-2774">2774</a>
<a href="#cl-2775">2775</a>
<a href="#cl-2776">2776</a>
<a href="#cl-2777">2777</a>
<a href="#cl-2778">2778</a>
<a href="#cl-2779">2779</a>
<a href="#cl-2780">2780</a>
<a href="#cl-2781">2781</a>
<a href="#cl-2782">2782</a>
<a href="#cl-2783">2783</a>
<a href="#cl-2784">2784</a>
<a href="#cl-2785">2785</a>
<a href="#cl-2786">2786</a>
<a href="#cl-2787">2787</a>
<a href="#cl-2788">2788</a>
<a href="#cl-2789">2789</a>
<a href="#cl-2790">2790</a>
<a href="#cl-2791">2791</a>
<a href="#cl-2792">2792</a>
<a href="#cl-2793">2793</a>
<a href="#cl-2794">2794</a>
<a href="#cl-2795">2795</a>
<a href="#cl-2796">2796</a>
<a href="#cl-2797">2797</a>
<a href="#cl-2798">2798</a>
<a href="#cl-2799">2799</a>
<a href="#cl-2800">2800</a>
<a href="#cl-2801">2801</a>
<a href="#cl-2802">2802</a>
<a href="#cl-2803">2803</a>
<a href="#cl-2804">2804</a>
<a href="#cl-2805">2805</a>
<a href="#cl-2806">2806</a>
<a href="#cl-2807">2807</a>
<a href="#cl-2808">2808</a>
<a href="#cl-2809">2809</a>
<a href="#cl-2810">2810</a>
<a href="#cl-2811">2811</a>
<a href="#cl-2812">2812</a>
<a href="#cl-2813">2813</a>
<a href="#cl-2814">2814</a>
<a href="#cl-2815">2815</a>
<a href="#cl-2816">2816</a>
<a href="#cl-2817">2817</a>
<a href="#cl-2818">2818</a>
<a href="#cl-2819">2819</a>
<a href="#cl-2820">2820</a>
<a href="#cl-2821">2821</a>
<a href="#cl-2822">2822</a>
<a href="#cl-2823">2823</a>
<a href="#cl-2824">2824</a>
<a href="#cl-2825">2825</a>
<a href="#cl-2826">2826</a>
<a href="#cl-2827">2827</a>
<a href="#cl-2828">2828</a>
<a href="#cl-2829">2829</a>
<a href="#cl-2830">2830</a>
<a href="#cl-2831">2831</a>
<a href="#cl-2832">2832</a>
<a href="#cl-2833">2833</a>
<a href="#cl-2834">2834</a>
<a href="#cl-2835">2835</a>
<a href="#cl-2836">2836</a>
<a href="#cl-2837">2837</a>
<a href="#cl-2838">2838</a>
<a href="#cl-2839">2839</a>
<a href="#cl-2840">2840</a>
<a href="#cl-2841">2841</a>
<a href="#cl-2842">2842</a>
<a href="#cl-2843">2843</a>
<a href="#cl-2844">2844</a>
<a href="#cl-2845">2845</a>
<a href="#cl-2846">2846</a>
<a href="#cl-2847">2847</a>
<a href="#cl-2848">2848</a>
<a href="#cl-2849">2849</a>
<a href="#cl-2850">2850</a>
<a href="#cl-2851">2851</a>
<a href="#cl-2852">2852</a>
<a href="#cl-2853">2853</a>
<a href="#cl-2854">2854</a>
<a href="#cl-2855">2855</a>
<a href="#cl-2856">2856</a>
<a href="#cl-2857">2857</a>
<a href="#cl-2858">2858</a>
<a href="#cl-2859">2859</a>
<a href="#cl-2860">2860</a>
<a href="#cl-2861">2861</a>
<a href="#cl-2862">2862</a>
<a href="#cl-2863">2863</a>
<a href="#cl-2864">2864</a>
<a href="#cl-2865">2865</a>
<a href="#cl-2866">2866</a>
<a href="#cl-2867">2867</a>
<a href="#cl-2868">2868</a>
<a href="#cl-2869">2869</a>
<a href="#cl-2870">2870</a>
<a href="#cl-2871">2871</a>
<a href="#cl-2872">2872</a>
<a href="#cl-2873">2873</a>
<a href="#cl-2874">2874</a>
<a href="#cl-2875">2875</a>
<a href="#cl-2876">2876</a>
<a href="#cl-2877">2877</a>
<a href="#cl-2878">2878</a>
<a href="#cl-2879">2879</a>
<a href="#cl-2880">2880</a>
<a href="#cl-2881">2881</a>
<a href="#cl-2882">2882</a>
<a href="#cl-2883">2883</a>
<a href="#cl-2884">2884</a>
<a href="#cl-2885">2885</a>
<a href="#cl-2886">2886</a>
<a href="#cl-2887">2887</a>
<a href="#cl-2888">2888</a>
<a href="#cl-2889">2889</a>
<a href="#cl-2890">2890</a>
<a href="#cl-2891">2891</a>
<a href="#cl-2892">2892</a>
<a href="#cl-2893">2893</a>
<a href="#cl-2894">2894</a>
<a href="#cl-2895">2895</a>
<a href="#cl-2896">2896</a>
<a href="#cl-2897">2897</a>
<a href="#cl-2898">2898</a>
<a href="#cl-2899">2899</a>
<a href="#cl-2900">2900</a>
<a href="#cl-2901">2901</a>
<a href="#cl-2902">2902</a>
<a href="#cl-2903">2903</a>
<a href="#cl-2904">2904</a>
<a href="#cl-2905">2905</a>
<a href="#cl-2906">2906</a>
<a href="#cl-2907">2907</a>
<a href="#cl-2908">2908</a>
<a href="#cl-2909">2909</a>
<a href="#cl-2910">2910</a>
<a href="#cl-2911">2911</a>
<a href="#cl-2912">2912</a>
<a href="#cl-2913">2913</a>
<a href="#cl-2914">2914</a>
<a href="#cl-2915">2915</a>
<a href="#cl-2916">2916</a>
<a href="#cl-2917">2917</a>
<a href="#cl-2918">2918</a>
<a href="#cl-2919">2919</a>
<a href="#cl-2920">2920</a>
<a href="#cl-2921">2921</a>
<a href="#cl-2922">2922</a>
<a href="#cl-2923">2923</a>
<a href="#cl-2924">2924</a>
<a href="#cl-2925">2925</a>
<a href="#cl-2926">2926</a>
<a href="#cl-2927">2927</a>
<a href="#cl-2928">2928</a>
<a href="#cl-2929">2929</a>
<a href="#cl-2930">2930</a>
<a href="#cl-2931">2931</a>
<a href="#cl-2932">2932</a>
<a href="#cl-2933">2933</a>
<a href="#cl-2934">2934</a>
<a href="#cl-2935">2935</a>
<a href="#cl-2936">2936</a>
<a href="#cl-2937">2937</a>
<a href="#cl-2938">2938</a>
<a href="#cl-2939">2939</a>
<a href="#cl-2940">2940</a>
<a href="#cl-2941">2941</a>
<a href="#cl-2942">2942</a>
<a href="#cl-2943">2943</a>
<a href="#cl-2944">2944</a>
<a href="#cl-2945">2945</a>
<a href="#cl-2946">2946</a>
<a href="#cl-2947">2947</a>
<a href="#cl-2948">2948</a>
<a href="#cl-2949">2949</a>
<a href="#cl-2950">2950</a>
<a href="#cl-2951">2951</a>
<a href="#cl-2952">2952</a>
<a href="#cl-2953">2953</a>
<a href="#cl-2954">2954</a>
<a href="#cl-2955">2955</a>
<a href="#cl-2956">2956</a>
<a href="#cl-2957">2957</a>
<a href="#cl-2958">2958</a>
<a href="#cl-2959">2959</a>
<a href="#cl-2960">2960</a>
<a href="#cl-2961">2961</a>
<a href="#cl-2962">2962</a>
<a href="#cl-2963">2963</a>
<a href="#cl-2964">2964</a>
<a href="#cl-2965">2965</a>
<a href="#cl-2966">2966</a>
<a href="#cl-2967">2967</a>
<a href="#cl-2968">2968</a>
<a href="#cl-2969">2969</a>
<a href="#cl-2970">2970</a>
<a href="#cl-2971">2971</a>
<a href="#cl-2972">2972</a>
<a href="#cl-2973">2973</a>
<a href="#cl-2974">2974</a>
<a href="#cl-2975">2975</a>
<a href="#cl-2976">2976</a>
<a href="#cl-2977">2977</a>
<a href="#cl-2978">2978</a>
<a href="#cl-2979">2979</a>
<a href="#cl-2980">2980</a>
<a href="#cl-2981">2981</a>
<a href="#cl-2982">2982</a>
<a href="#cl-2983">2983</a>
<a href="#cl-2984">2984</a>
<a href="#cl-2985">2985</a>
<a href="#cl-2986">2986</a>
<a href="#cl-2987">2987</a>
<a href="#cl-2988">2988</a>
<a href="#cl-2989">2989</a>
<a href="#cl-2990">2990</a>
<a href="#cl-2991">2991</a>
<a href="#cl-2992">2992</a>
<a href="#cl-2993">2993</a>
<a href="#cl-2994">2994</a>
<a href="#cl-2995">2995</a>
<a href="#cl-2996">2996</a>
<a href="#cl-2997">2997</a>
<a href="#cl-2998">2998</a>
<a href="#cl-2999">2999</a>
<a href="#cl-3000">3000</a>
<a href="#cl-3001">3001</a>
<a href="#cl-3002">3002</a>
<a href="#cl-3003">3003</a>
<a href="#cl-3004">3004</a>
<a href="#cl-3005">3005</a>
<a href="#cl-3006">3006</a>
<a href="#cl-3007">3007</a>
<a href="#cl-3008">3008</a>
<a href="#cl-3009">3009</a>
<a href="#cl-3010">3010</a>
<a href="#cl-3011">3011</a>
<a href="#cl-3012">3012</a>
<a href="#cl-3013">3013</a>
<a href="#cl-3014">3014</a>
<a href="#cl-3015">3015</a>
<a href="#cl-3016">3016</a>
<a href="#cl-3017">3017</a>
<a href="#cl-3018">3018</a>
<a href="#cl-3019">3019</a>
<a href="#cl-3020">3020</a>
<a href="#cl-3021">3021</a>
<a href="#cl-3022">3022</a>
<a href="#cl-3023">3023</a>
<a href="#cl-3024">3024</a>
<a href="#cl-3025">3025</a>
<a href="#cl-3026">3026</a>
<a href="#cl-3027">3027</a>
<a href="#cl-3028">3028</a>
<a href="#cl-3029">3029</a>
<a href="#cl-3030">3030</a>
<a href="#cl-3031">3031</a>
<a href="#cl-3032">3032</a>
<a href="#cl-3033">3033</a>
<a href="#cl-3034">3034</a>
<a href="#cl-3035">3035</a>
<a href="#cl-3036">3036</a>
<a href="#cl-3037">3037</a>
<a href="#cl-3038">3038</a>
<a href="#cl-3039">3039</a>
<a href="#cl-3040">3040</a>
<a href="#cl-3041">3041</a>
<a href="#cl-3042">3042</a>
<a href="#cl-3043">3043</a>
<a href="#cl-3044">3044</a>
<a href="#cl-3045">3045</a>
<a href="#cl-3046">3046</a>
<a href="#cl-3047">3047</a>
<a href="#cl-3048">3048</a>
<a href="#cl-3049">3049</a>
<a href="#cl-3050">3050</a>
<a href="#cl-3051">3051</a>
<a href="#cl-3052">3052</a>
<a href="#cl-3053">3053</a>
<a href="#cl-3054">3054</a>
<a href="#cl-3055">3055</a>
<a href="#cl-3056">3056</a>
<a href="#cl-3057">3057</a>
<a href="#cl-3058">3058</a>
<a href="#cl-3059">3059</a>
<a href="#cl-3060">3060</a>
<a href="#cl-3061">3061</a>
<a href="#cl-3062">3062</a>
<a href="#cl-3063">3063</a>
<a href="#cl-3064">3064</a>
<a href="#cl-3065">3065</a>
<a href="#cl-3066">3066</a>
<a href="#cl-3067">3067</a>
<a href="#cl-3068">3068</a>
<a href="#cl-3069">3069</a>
<a href="#cl-3070">3070</a>
<a href="#cl-3071">3071</a>
<a href="#cl-3072">3072</a>
<a href="#cl-3073">3073</a>
<a href="#cl-3074">3074</a>
<a href="#cl-3075">3075</a>
<a href="#cl-3076">3076</a>
<a href="#cl-3077">3077</a>
<a href="#cl-3078">3078</a>
<a href="#cl-3079">3079</a>
<a href="#cl-3080">3080</a>
<a href="#cl-3081">3081</a>
<a href="#cl-3082">3082</a>
<a href="#cl-3083">3083</a>
<a href="#cl-3084">3084</a>
<a href="#cl-3085">3085</a>
<a href="#cl-3086">3086</a>
<a href="#cl-3087">3087</a>
<a href="#cl-3088">3088</a>
<a href="#cl-3089">3089</a>
<a href="#cl-3090">3090</a>
<a href="#cl-3091">3091</a>
<a href="#cl-3092">3092</a>
<a href="#cl-3093">3093</a>
<a href="#cl-3094">3094</a>
<a href="#cl-3095">3095</a>
<a href="#cl-3096">3096</a>
<a href="#cl-3097">3097</a>
<a href="#cl-3098">3098</a>
<a href="#cl-3099">3099</a>
<a href="#cl-3100">3100</a>
<a href="#cl-3101">3101</a>
<a href="#cl-3102">3102</a>
<a href="#cl-3103">3103</a>
<a href="#cl-3104">3104</a>
<a href="#cl-3105">3105</a>
<a href="#cl-3106">3106</a>
<a href="#cl-3107">3107</a>
<a href="#cl-3108">3108</a>
<a href="#cl-3109">3109</a>
<a href="#cl-3110">3110</a>
<a href="#cl-3111">3111</a>
<a href="#cl-3112">3112</a>
<a href="#cl-3113">3113</a>
<a href="#cl-3114">3114</a>
<a href="#cl-3115">3115</a>
<a href="#cl-3116">3116</a>
<a href="#cl-3117">3117</a>
<a href="#cl-3118">3118</a>
<a href="#cl-3119">3119</a>
<a href="#cl-3120">3120</a>
<a href="#cl-3121">3121</a>
<a href="#cl-3122">3122</a>
<a href="#cl-3123">3123</a>
<a href="#cl-3124">3124</a>
<a href="#cl-3125">3125</a>
<a href="#cl-3126">3126</a>
<a href="#cl-3127">3127</a>
<a href="#cl-3128">3128</a>
<a href="#cl-3129">3129</a>
<a href="#cl-3130">3130</a>
<a href="#cl-3131">3131</a>
<a href="#cl-3132">3132</a>
<a href="#cl-3133">3133</a>
<a href="#cl-3134">3134</a>
<a href="#cl-3135">3135</a>
<a href="#cl-3136">3136</a>
<a href="#cl-3137">3137</a>
<a href="#cl-3138">3138</a>
<a href="#cl-3139">3139</a>
<a href="#cl-3140">3140</a>
<a href="#cl-3141">3141</a>
<a href="#cl-3142">3142</a>
<a href="#cl-3143">3143</a>
<a href="#cl-3144">3144</a>
<a href="#cl-3145">3145</a>
<a href="#cl-3146">3146</a>
<a href="#cl-3147">3147</a>
<a href="#cl-3148">3148</a>
<a href="#cl-3149">3149</a>
<a href="#cl-3150">3150</a>
<a href="#cl-3151">3151</a>
<a href="#cl-3152">3152</a>
<a href="#cl-3153">3153</a>
<a href="#cl-3154">3154</a>
<a href="#cl-3155">3155</a>
<a href="#cl-3156">3156</a>
<a href="#cl-3157">3157</a>
<a href="#cl-3158">3158</a>
<a href="#cl-3159">3159</a>
<a href="#cl-3160">3160</a>
<a href="#cl-3161">3161</a>
<a href="#cl-3162">3162</a>
<a href="#cl-3163">3163</a>
<a href="#cl-3164">3164</a>
<a href="#cl-3165">3165</a>
<a href="#cl-3166">3166</a>
<a href="#cl-3167">3167</a>
<a href="#cl-3168">3168</a>
<a href="#cl-3169">3169</a>
<a href="#cl-3170">3170</a>
<a href="#cl-3171">3171</a>
<a href="#cl-3172">3172</a>
<a href="#cl-3173">3173</a>
<a href="#cl-3174">3174</a>
<a href="#cl-3175">3175</a>
<a href="#cl-3176">3176</a>
<a href="#cl-3177">3177</a>
<a href="#cl-3178">3178</a>
<a href="#cl-3179">3179</a>
<a href="#cl-3180">3180</a>
<a href="#cl-3181">3181</a>
<a href="#cl-3182">3182</a>
<a href="#cl-3183">3183</a>
<a href="#cl-3184">3184</a>
<a href="#cl-3185">3185</a>
<a href="#cl-3186">3186</a>
<a href="#cl-3187">3187</a>
<a href="#cl-3188">3188</a>
<a href="#cl-3189">3189</a>
<a href="#cl-3190">3190</a>
<a href="#cl-3191">3191</a>
<a href="#cl-3192">3192</a>
<a href="#cl-3193">3193</a>
<a href="#cl-3194">3194</a>
<a href="#cl-3195">3195</a>
<a href="#cl-3196">3196</a>
<a href="#cl-3197">3197</a>
<a href="#cl-3198">3198</a>
<a href="#cl-3199">3199</a>
<a href="#cl-3200">3200</a>
<a href="#cl-3201">3201</a>
<a href="#cl-3202">3202</a>
<a href="#cl-3203">3203</a>
<a href="#cl-3204">3204</a>
<a href="#cl-3205">3205</a>
<a href="#cl-3206">3206</a>
<a href="#cl-3207">3207</a>
<a href="#cl-3208">3208</a>
<a href="#cl-3209">3209</a>
<a href="#cl-3210">3210</a>
<a href="#cl-3211">3211</a>
<a href="#cl-3212">3212</a>
<a href="#cl-3213">3213</a>
<a href="#cl-3214">3214</a>
<a href="#cl-3215">3215</a>
<a href="#cl-3216">3216</a>
</pre></div></td>
<td class="code"><div class="highlight"><pre>
<a name="cl-1"></a># cython: profile=False
<a name="cl-2"></a>
<a name="cl-3"></a>import logging
<a name="cl-4"></a>import os
<a name="cl-5"></a>import collections
<a name="cl-6"></a>import sys
<a name="cl-7"></a>import gc
<a name="cl-8"></a>
<a name="cl-9"></a>import numpy
<a name="cl-10"></a>cimport numpy
<a name="cl-11"></a>cimport cython
<a name="cl-12"></a>import osgeo
<a name="cl-13"></a>from osgeo import gdal
<a name="cl-14"></a>from cython.operator cimport dereference as deref
<a name="cl-15"></a>
<a name="cl-16"></a>from libcpp.set cimport set as c_set
<a name="cl-17"></a>from libcpp.deque cimport deque
<a name="cl-18"></a>from libcpp.map cimport map
<a name="cl-19"></a>from libcpp.stack cimport stack
<a name="cl-20"></a>from libc.math cimport atan
<a name="cl-21"></a>from libc.math cimport atan2
<a name="cl-22"></a>from libc.math cimport tan
<a name="cl-23"></a>from libc.math cimport sqrt
<a name="cl-24"></a>from libc.math cimport ceil
<a name="cl-25"></a>
<a name="cl-26"></a>cdef extern from "time.h" nogil:
<a name="cl-27"></a>    ctypedef int time_t
<a name="cl-28"></a>    time_t time(time_t*)
<a name="cl-29"></a>
<a name="cl-30"></a>import pygeoprocessing
<a name="cl-31"></a>
<a name="cl-32"></a>logging.basicConfig(format='%(asctime)s %(name)-18s %(levelname)-8s \
<a name="cl-33"></a>    %(message)s', lnevel=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')
<a name="cl-34"></a>
<a name="cl-35"></a>LOGGER = logging.getLogger('pygeoprocessing.routing.routing_core')
<a name="cl-36"></a>
<a name="cl-37"></a>cdef int N_MONTHS = 12
<a name="cl-38"></a>
<a name="cl-39"></a>cdef double PI = 3.141592653589793238462643383279502884
<a name="cl-40"></a>cdef double INF = numpy.inf
<a name="cl-41"></a>cdef int N_BLOCK_ROWS = 6
<a name="cl-42"></a>cdef int N_BLOCK_COLS = 6
<a name="cl-43"></a>
<a name="cl-44"></a>cdef class BlockCache_SWY:
<a name="cl-45"></a>    cdef numpy.int32_t[:,:] row_tag_cache
<a name="cl-46"></a>    cdef numpy.int32_t[:,:] col_tag_cache
<a name="cl-47"></a>    cdef numpy.int8_t[:,:] cache_dirty
<a name="cl-48"></a>    cdef int n_block_rows
<a name="cl-49"></a>    cdef int n_block_cols
<a name="cl-50"></a>    cdef int block_col_size
<a name="cl-51"></a>    cdef int block_row_size
<a name="cl-52"></a>    cdef int n_rows
<a name="cl-53"></a>    cdef int n_cols
<a name="cl-54"></a>    band_list = []
<a name="cl-55"></a>    block_list = []
<a name="cl-56"></a>    update_list = []
<a name="cl-57"></a>
<a name="cl-58"></a>    def __cinit__(
<a name="cl-59"></a>            self, int n_block_rows, int n_block_cols, int n_rows, int n_cols,
<a name="cl-60"></a>            int block_row_size, int block_col_size, band_list, block_list,
<a name="cl-61"></a>            update_list, numpy.int8_t[:,:] cache_dirty):
<a name="cl-62"></a>        self.n_block_rows = n_block_rows
<a name="cl-63"></a>        self.n_block_cols = n_block_cols
<a name="cl-64"></a>        self.block_col_size = block_col_size
<a name="cl-65"></a>        self.block_row_size = block_row_size
<a name="cl-66"></a>        self.n_rows = n_rows
<a name="cl-67"></a>        self.n_cols = n_cols
<a name="cl-68"></a>        self.row_tag_cache = numpy.zeros((n_block_rows, n_block_cols), dtype=numpy.int32)
<a name="cl-69"></a>        self.col_tag_cache = numpy.zeros((n_block_rows, n_block_cols), dtype=numpy.int32)
<a name="cl-70"></a>        self.cache_dirty = cache_dirty
<a name="cl-71"></a>        self.row_tag_cache[:] = -1
<a name="cl-72"></a>        self.col_tag_cache[:] = -1
<a name="cl-73"></a>        self.band_list[:] = band_list
<a name="cl-74"></a>        self.block_list[:] = block_list
<a name="cl-75"></a>        self.update_list[:] = update_list
<a name="cl-76"></a>        list_lengths = [len(x) for x in [band_list, block_list, update_list]]
<a name="cl-77"></a>        if len(set(list_lengths)) &gt; 1:
<a name="cl-78"></a>            raise ValueError(
<a name="cl-79"></a>                "lengths of band_list, block_list, update_list should be equal."
<a name="cl-80"></a>                " instead they are %s", list_lengths)
<a name="cl-81"></a>        raster_dimensions_list = [(b.YSize, b.XSize) for b in band_list]
<a name="cl-82"></a>        for raster_n_rows, raster_n_cols in raster_dimensions_list:
<a name="cl-83"></a>            if raster_n_rows != n_rows or raster_n_cols != n_cols:
<a name="cl-84"></a>                raise ValueError(
<a name="cl-85"></a>                    "a band was passed in that has a different dimension than"
<a name="cl-86"></a>                    "the memory block was specified as")
<a name="cl-87"></a>
<a name="cl-88"></a>        for band in band_list:
<a name="cl-89"></a>            block_col_size, block_row_size = band.GetBlockSize()
<a name="cl-90"></a>            if block_col_size == 1 or block_row_size == 1:
<a name="cl-91"></a>                LOGGER.warn(
<a name="cl-92"></a>                    'a band in BlockCache is not memory blocked, this might '
<a name="cl-93"></a>                    'make the runtime slow for other algorithms. %s',
<a name="cl-94"></a>                    band.GetDescription())
<a name="cl-95"></a>
<a name="cl-96"></a>    def __dealloc__(self):
<a name="cl-97"></a>        self.band_list[:] = []
<a name="cl-98"></a>        self.block_list[:] = []
<a name="cl-99"></a>        self.update_list[:] = []
<a name="cl-100"></a>
<a name="cl-101"></a>    #@cython.boundscheck(False)
<a name="cl-102"></a>    @cython.wraparound(False)
<a name="cl-103"></a>    #@cython.cdivision(True)
<a name="cl-104"></a>    cdef void update_cache(self, int global_row, int global_col, int *row_index, int *col_index, int *row_block_offset, int *col_block_offset):
<a name="cl-105"></a>        cdef int cache_row_size, cache_col_size
<a name="cl-106"></a>        cdef int global_row_offset, global_col_offset
<a name="cl-107"></a>        cdef int row_tag, col_tag
<a name="cl-108"></a>
<a name="cl-109"></a>        row_block_offset[0] = global_row % self.block_row_size
<a name="cl-110"></a>        row_index[0] = (global_row // self.block_row_size) % self.n_block_rows
<a name="cl-111"></a>        row_tag = (global_row // self.block_row_size) // self.n_block_rows
<a name="cl-112"></a>
<a name="cl-113"></a>        col_block_offset[0] = global_col % self.block_col_size
<a name="cl-114"></a>        col_index[0] = (global_col // self.block_col_size) % self.n_block_cols
<a name="cl-115"></a>        col_tag = (global_col // self.block_col_size) // self.n_block_cols
<a name="cl-116"></a>
<a name="cl-117"></a>        cdef int current_row_tag = self.row_tag_cache[row_index[0], col_index[0]]
<a name="cl-118"></a>        cdef int current_col_tag = self.col_tag_cache[row_index[0], col_index[0]]
<a name="cl-119"></a>
<a name="cl-120"></a>        if current_row_tag != row_tag or current_col_tag != col_tag:
<a name="cl-121"></a>            if self.cache_dirty[row_index[0], col_index[0]]:
<a name="cl-122"></a>                global_col_offset = (current_col_tag * self.n_block_cols + col_index[0]) * self.block_col_size
<a name="cl-123"></a>                cache_col_size = self.n_cols - global_col_offset
<a name="cl-124"></a>                if cache_col_size &gt; self.block_col_size:
<a name="cl-125"></a>                    cache_col_size = self.block_col_size
<a name="cl-126"></a>
<a name="cl-127"></a>                global_row_offset = (current_row_tag * self.n_block_rows + row_index[0]) * self.block_row_size
<a name="cl-128"></a>                cache_row_size = self.n_rows - global_row_offset
<a name="cl-129"></a>                if cache_row_size &gt; self.block_row_size:
<a name="cl-130"></a>                    cache_row_size = self.block_row_size
<a name="cl-131"></a>
<a name="cl-132"></a>                for band, block, update in zip(self.band_list, self.block_list, self.update_list):
<a name="cl-133"></a>                    if update:
<a name="cl-134"></a>                        band.WriteArray(block[row_index[0], col_index[0], 0:cache_row_size, 0:cache_col_size],
<a name="cl-135"></a>                            yoff=global_row_offset, xoff=global_col_offset)
<a name="cl-136"></a>                self.cache_dirty[row_index[0], col_index[0]] = 0
<a name="cl-137"></a>            self.row_tag_cache[row_index[0], col_index[0]] = row_tag
<a name="cl-138"></a>            self.col_tag_cache[row_index[0], col_index[0]] = col_tag
<a name="cl-139"></a>
<a name="cl-140"></a>            global_col_offset = (col_tag * self.n_block_cols + col_index[0]) * self.block_col_size
<a name="cl-141"></a>            global_row_offset = (row_tag * self.n_block_rows + row_index[0]) * self.block_row_size
<a name="cl-142"></a>
<a name="cl-143"></a>            cache_col_size = self.n_cols - global_col_offset
<a name="cl-144"></a>            if cache_col_size &gt; self.block_col_size:
<a name="cl-145"></a>                cache_col_size = self.block_col_size
<a name="cl-146"></a>            cache_row_size = self.n_rows - global_row_offset
<a name="cl-147"></a>            if cache_row_size &gt; self.block_row_size:
<a name="cl-148"></a>                cache_row_size = self.block_row_size
<a name="cl-149"></a>
<a name="cl-150"></a>            for band_index, (band, block) in enumerate(zip(self.band_list, self.block_list)):
<a name="cl-151"></a>                band.ReadAsArray(
<a name="cl-152"></a>                    xoff=global_col_offset, yoff=global_row_offset,
<a name="cl-153"></a>                    win_xsize=cache_col_size, win_ysize=cache_row_size,
<a name="cl-154"></a>                    buf_obj=block[row_index[0], col_index[0], 0:cache_row_size, 0:cache_col_size])
<a name="cl-155"></a>
<a name="cl-156"></a>    cdef void flush_cache(self):
<a name="cl-157"></a>        cdef int global_row_offset, global_col_offset
<a name="cl-158"></a>        cdef int cache_row_size, cache_col_size
<a name="cl-159"></a>        cdef int row_index, col_index
<a name="cl-160"></a>        for row_index in xrange(self.n_block_rows):
<a name="cl-161"></a>            for col_index in xrange(self.n_block_cols):
<a name="cl-162"></a>                row_tag = self.row_tag_cache[row_index, col_index]
<a name="cl-163"></a>                col_tag = self.col_tag_cache[row_index, col_index]
<a name="cl-164"></a>
<a name="cl-165"></a>                if self.cache_dirty[row_index, col_index]:
<a name="cl-166"></a>                    global_col_offset = (col_tag * self.n_block_cols + col_index) * self.block_col_size
<a name="cl-167"></a>                    cache_col_size = self.n_cols - global_col_offset
<a name="cl-168"></a>                    if cache_col_size &gt; self.block_col_size:
<a name="cl-169"></a>                        cache_col_size = self.block_col_size
<a name="cl-170"></a>
<a name="cl-171"></a>                    global_row_offset = (row_tag * self.n_block_rows + row_index) * self.block_row_size
<a name="cl-172"></a>                    cache_row_size = self.n_rows - global_row_offset
<a name="cl-173"></a>                    if cache_row_size &gt; self.block_row_size:
<a name="cl-174"></a>                        cache_row_size = self.block_row_size
<a name="cl-175"></a>
<a name="cl-176"></a>                    for band, block, update in zip(self.band_list, self.block_list, self.update_list):
<a name="cl-177"></a>                        if update:
<a name="cl-178"></a>                            band.WriteArray(block[row_index, col_index, 0:cache_row_size, 0:cache_col_size],
<a name="cl-179"></a>                                yoff=global_row_offset, xoff=global_col_offset)
<a name="cl-180"></a>        for band in self.band_list:
<a name="cl-181"></a>            band.FlushCache()
<a name="cl-182"></a>
<a name="cl-183"></a>#@cython.boundscheck(False)
<a name="cl-184"></a>@cython.wraparound(False)
<a name="cl-185"></a>cdef route_recharge(
<a name="cl-186"></a>        precip_uri_list, et0_uri_list, kc_uri, recharge_uri, recharge_avail_uri,
<a name="cl-187"></a>        r_sum_avail_uri, aet_uri, float alpha_m, float beta_i, float gamma,
<a name="cl-188"></a>        qfi_uri_list, outflow_direction_uri, outflow_weights_uri, stream_uri,
<a name="cl-189"></a>        deque[int] &amp;sink_cell_deque):
<a name="cl-190"></a>
<a name="cl-191"></a>    #Pass transport
<a name="cl-192"></a>    cdef time_t start
<a name="cl-193"></a>    time(&amp;start)
<a name="cl-194"></a>
<a name="cl-195"></a>    #load a base dataset so we can determine the n_rows/cols
<a name="cl-196"></a>    outflow_direction_dataset = gdal.Open(outflow_direction_uri)
<a name="cl-197"></a>    cdef int n_cols = outflow_direction_dataset.RasterXSize
<a name="cl-198"></a>    cdef int n_rows = outflow_direction_dataset.RasterYSize
<a name="cl-199"></a>    outflow_direction_band = outflow_direction_dataset.GetRasterBand(1)
<a name="cl-200"></a>
<a name="cl-201"></a>    cdef int block_col_size, block_row_size
<a name="cl-202"></a>    block_col_size, block_row_size = outflow_direction_band.GetBlockSize()
<a name="cl-203"></a>
<a name="cl-204"></a>    #center point of global index
<a name="cl-205"></a>    cdef int global_row, global_col #index into the overall raster
<a name="cl-206"></a>    cdef int row_index, col_index #the index of the cache block
<a name="cl-207"></a>    cdef int row_block_offset, col_block_offset #index into the cache block
<a name="cl-208"></a>    cdef int global_block_row, global_block_col #used to walk the global blocks
<a name="cl-209"></a>
<a name="cl-210"></a>    #neighbor sections of global index
<a name="cl-211"></a>    cdef int neighbor_row, neighbor_col #neighbor equivalent of global_{row,col}
<a name="cl-212"></a>    cdef int neighbor_row_index, neighbor_col_index #neighbor cache index
<a name="cl-213"></a>    cdef int neighbor_row_block_offset, neighbor_col_block_offset #index into the neighbor cache block
<a name="cl-214"></a>
<a name="cl-215"></a>    #define all the single caches
<a name="cl-216"></a>    cdef numpy.ndarray[numpy.npy_int8, ndim=4] outflow_direction_block = numpy.zeros(
<a name="cl-217"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.int8)
<a name="cl-218"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] outflow_weights_block = numpy.zeros(
<a name="cl-219"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-220"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] kc_block = numpy.zeros(
<a name="cl-221"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-222"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] recharge_block = numpy.zeros(
<a name="cl-223"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-224"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] recharge_avail_block = numpy.zeros(
<a name="cl-225"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-226"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] r_sum_avail_block = numpy.zeros(
<a name="cl-227"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-228"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] aet_block = numpy.zeros(
<a name="cl-229"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-230"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] stream_block = numpy.zeros(
<a name="cl-231"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
<a name="cl-232"></a>        dtype=numpy.float32)
<a name="cl-233"></a>
<a name="cl-234"></a>
<a name="cl-235"></a>    #these are 12 band blocks
<a name="cl-236"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=5] precip_block_list = numpy.zeros(
<a name="cl-237"></a>        (N_MONTHS, N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-238"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=5] et0_block_list = numpy.zeros(
<a name="cl-239"></a>        (N_MONTHS, N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-240"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=5] qfi_block_list = numpy.zeros(
<a name="cl-241"></a>        (N_MONTHS, N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-242"></a>
<a name="cl-243"></a>    cdef numpy.ndarray[numpy.npy_int8, ndim=2] cache_dirty = numpy.zeros(
<a name="cl-244"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.int8)
<a name="cl-245"></a>
<a name="cl-246"></a>    cdef int outflow_direction_nodata = pygeoprocessing.get_nodata_from_uri(
<a name="cl-247"></a>        outflow_direction_uri)
<a name="cl-248"></a>
<a name="cl-249"></a>    #load the et0 and precip bands
<a name="cl-250"></a>    et0_dataset_list = []
<a name="cl-251"></a>    et0_band_list = []
<a name="cl-252"></a>    precip_datset_list = []
<a name="cl-253"></a>    precip_band_list = []
<a name="cl-254"></a>
<a name="cl-255"></a>    for uri_list, dataset_list, band_list in [
<a name="cl-256"></a>            (et0_uri_list, et0_dataset_list, et0_band_list),
<a name="cl-257"></a>            (precip_uri_list, precip_datset_list, precip_band_list)]:
<a name="cl-258"></a>        for index, uri in enumerate(uri_list):
<a name="cl-259"></a>            dataset_list.append(gdal.Open(uri))
<a name="cl-260"></a>            band_list.append(dataset_list[index].GetRasterBand(1))
<a name="cl-261"></a>
<a name="cl-262"></a>    cdef float precip_nodata = pygeoprocessing.get_nodata_from_uri(precip_uri_list[0])
<a name="cl-263"></a>    cdef float et0_nodata = pygeoprocessing.get_nodata_from_uri(et0_uri_list[0])
<a name="cl-264"></a>
<a name="cl-265"></a>    qfi_datset_list = []
<a name="cl-266"></a>    qfi_band_list = []
<a name="cl-267"></a>
<a name="cl-268"></a>    outflow_weights_dataset = gdal.Open(outflow_weights_uri)
<a name="cl-269"></a>    outflow_weights_band = outflow_weights_dataset.GetRasterBand(1)
<a name="cl-270"></a>    cdef float outflow_weights_nodata = pygeoprocessing.get_nodata_from_uri(
<a name="cl-271"></a>        outflow_weights_uri)
<a name="cl-272"></a>    kc_dataset = gdal.Open(kc_uri)
<a name="cl-273"></a>    kc_band = kc_dataset.GetRasterBand(1)
<a name="cl-274"></a>    cdef float kc_nodata = pygeoprocessing.get_nodata_from_uri(
<a name="cl-275"></a>        kc_uri)
<a name="cl-276"></a>    stream_dataset = gdal.Open(stream_uri)
<a name="cl-277"></a>    stream_band = stream_dataset.GetRasterBand(1)
<a name="cl-278"></a>
<a name="cl-279"></a>    #Create output arrays qfi and recharge and recharge_avail
<a name="cl-280"></a>    cdef float recharge_nodata = -99999
<a name="cl-281"></a>    pygeoprocessing.new_raster_from_base_uri(
<a name="cl-282"></a>        outflow_direction_uri, recharge_uri, 'GTiff', recharge_nodata,
<a name="cl-283"></a>        gdal.GDT_Float32)
<a name="cl-284"></a>    recharge_dataset = gdal.Open(recharge_uri, gdal.GA_Update)
<a name="cl-285"></a>    recharge_band = recharge_dataset.GetRasterBand(1)
<a name="cl-286"></a>    pygeoprocessing.new_raster_from_base_uri(
<a name="cl-287"></a>        outflow_direction_uri, recharge_avail_uri, 'GTiff', recharge_nodata,
<a name="cl-288"></a>        gdal.GDT_Float32)
<a name="cl-289"></a>    recharge_avail_dataset = gdal.Open(recharge_avail_uri, gdal.GA_Update)
<a name="cl-290"></a>    recharge_avail_band = recharge_avail_dataset.GetRasterBand(1)
<a name="cl-291"></a>    pygeoprocessing.new_raster_from_base_uri(
<a name="cl-292"></a>        outflow_direction_uri, r_sum_avail_uri, 'GTiff', recharge_nodata,
<a name="cl-293"></a>        gdal.GDT_Float32)
<a name="cl-294"></a>    r_sum_avail_dataset = gdal.Open(r_sum_avail_uri, gdal.GA_Update)
<a name="cl-295"></a>    r_sum_avail_band = r_sum_avail_dataset.GetRasterBand(1)
<a name="cl-296"></a>
<a name="cl-297"></a>    cdef float aet_nodata = -99999
<a name="cl-298"></a>    pygeoprocessing.new_raster_from_base_uri(
<a name="cl-299"></a>        outflow_direction_uri, aet_uri, 'GTiff', aet_nodata,
<a name="cl-300"></a>        gdal.GDT_Float32)
<a name="cl-301"></a>    aet_dataset = gdal.Open(aet_uri, gdal.GA_Update)
<a name="cl-302"></a>    aet_band = aet_dataset.GetRasterBand(1)
<a name="cl-303"></a>
<a name="cl-304"></a>    qfi_dataset_list = []
<a name="cl-305"></a>    qfi_band_list = []
<a name="cl-306"></a>    cdef float qfi_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(
<a name="cl-307"></a>        qfi_uri_list[0])
<a name="cl-308"></a>    for index, qfi_uri in enumerate(qfi_uri_list):
<a name="cl-309"></a>        qfi_dataset_list.append(gdal.Open(qfi_uri, gdal.GA_ReadOnly))
<a name="cl-310"></a>        qfi_band_list.append(qfi_dataset_list[index].GetRasterBand(1))
<a name="cl-311"></a>
<a name="cl-312"></a>    band_list = ([
<a name="cl-313"></a>            outflow_direction_band,
<a name="cl-314"></a>            outflow_weights_band,
<a name="cl-315"></a>            kc_band,
<a name="cl-316"></a>            stream_band,
<a name="cl-317"></a>        ] + precip_band_list + et0_band_list + qfi_band_list +
<a name="cl-318"></a>        [recharge_band, recharge_avail_band, r_sum_avail_band, aet_band])
<a name="cl-319"></a>
<a name="cl-320"></a>    block_list = [outflow_direction_block, outflow_weights_block, kc_block, stream_block]
<a name="cl-321"></a>    block_list.extend([precip_block_list[i] for i in xrange(N_MONTHS)])
<a name="cl-322"></a>    block_list.extend([et0_block_list[i] for i in xrange(N_MONTHS)])
<a name="cl-323"></a>    block_list.extend([qfi_block_list[i] for i in xrange(N_MONTHS)])
<a name="cl-324"></a>    block_list.append(recharge_block)
<a name="cl-325"></a>    block_list.append(recharge_avail_block)
<a name="cl-326"></a>    block_list.append(r_sum_avail_block)
<a name="cl-327"></a>    block_list.append(aet_block)
<a name="cl-328"></a>
<a name="cl-329"></a>    update_list = (
<a name="cl-330"></a>        [False] * (4 + len(precip_band_list) + len(et0_band_list) + len(qfi_band_list)) +
<a name="cl-331"></a>        [True, True, True, True])
<a name="cl-332"></a>
<a name="cl-333"></a>    cache_dirty[:] = 0
<a name="cl-334"></a>
<a name="cl-335"></a>    cdef BlockCache_SWY block_cache = BlockCache_SWY(
<a name="cl-336"></a>        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols,
<a name="cl-337"></a>        block_row_size, block_col_size,
<a name="cl-338"></a>        band_list, block_list, update_list, cache_dirty)
<a name="cl-339"></a>
<a name="cl-340"></a>    #Process flux through the grid
<a name="cl-341"></a>    cdef stack[int] cells_to_process
<a name="cl-342"></a>    cdef stack[int] cell_neighbor_to_process
<a name="cl-343"></a>    cdef stack[float] r_sum_stack
<a name="cl-344"></a>
<a name="cl-345"></a>    for cell in sink_cell_deque:
<a name="cl-346"></a>        cells_to_process.push(cell)
<a name="cl-347"></a>        cell_neighbor_to_process.push(0)
<a name="cl-348"></a>        r_sum_stack.push(0.0)
<a name="cl-349"></a>
<a name="cl-350"></a>    #Diagonal offsets are based off the following index notation for neighbors
<a name="cl-351"></a>    #    3 2 1
<a name="cl-352"></a>    #    4 p 0
<a name="cl-353"></a>    #    5 6 7
<a name="cl-354"></a>
<a name="cl-355"></a>    cdef int *row_offsets = [0, -1, -1, -1,  0,  1, 1, 1]
<a name="cl-356"></a>    cdef int *col_offsets = [1,  1,  0, -1, -1, -1, 0, 1]
<a name="cl-357"></a>
<a name="cl-358"></a>    cdef int *inflow_offsets = [4, 5, 6, 7, 0, 1, 2, 3]
<a name="cl-359"></a>
<a name="cl-360"></a>    cdef int neighbor_direction
<a name="cl-361"></a>    cdef double absorption_rate
<a name="cl-362"></a>    cdef double outflow_weight
<a name="cl-363"></a>    cdef double in_flux
<a name="cl-364"></a>    cdef int current_neighbor_index
<a name="cl-365"></a>    cdef int current_index
<a name="cl-366"></a>    cdef float current_r_sum_avail
<a name="cl-367"></a>    cdef float qf_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(qfi_uri_list[0])
<a name="cl-368"></a>    cdef int month_index
<a name="cl-369"></a>    cdef float aet_sum
<a name="cl-370"></a>    cdef float pet_m
<a name="cl-371"></a>    cdef float aet_m
<a name="cl-372"></a>    cdef float p_i
<a name="cl-373"></a>    cdef float qf_i
<a name="cl-374"></a>    cdef float qfi_m
<a name="cl-375"></a>    cdef float p_m
<a name="cl-376"></a>    cdef float r_i
<a name="cl-377"></a>    cdef int neighbors_calculated = 0
<a name="cl-378"></a>
<a name="cl-379"></a>    cdef time_t last_time, current_time
<a name="cl-380"></a>    time(&amp;last_time)
<a name="cl-381"></a>    while not cells_to_process.empty():
<a name="cl-382"></a>        time(&amp;current_time)
<a name="cl-383"></a>        if current_time - last_time &gt; 5.0:
<a name="cl-384"></a>            LOGGER.info('route_recharge work queue size = %d' % (cells_to_process.size()))
<a name="cl-385"></a>            last_time = current_time
<a name="cl-386"></a>
<a name="cl-387"></a>        current_index = cells_to_process.top()
<a name="cl-388"></a>        cells_to_process.pop()
<a name="cl-389"></a>        with cython.cdivision(True):
<a name="cl-390"></a>            global_row = current_index / n_cols
<a name="cl-391"></a>            global_col = current_index % n_cols
<a name="cl-392"></a>        #see if we need to update the row cache
<a name="cl-393"></a>
<a name="cl-394"></a>        current_neighbor_index = cell_neighbor_to_process.top()
<a name="cl-395"></a>        cell_neighbor_to_process.pop()
<a name="cl-396"></a>        current_r_sum_avail = r_sum_stack.top()
<a name="cl-397"></a>        r_sum_stack.pop()
<a name="cl-398"></a>        neighbors_calculated = 1
<a name="cl-399"></a>
<a name="cl-400"></a>        block_cache.update_cache(global_row, global_col, &amp;row_index, &amp;col_index, &amp;row_block_offset, &amp;col_block_offset)
<a name="cl-401"></a>
<a name="cl-402"></a>        #Ensure we are working on a valid pixel, if not set everything to 0
<a name="cl-403"></a>            #check quickflow nodata? month 0? qfi_nodata
<a name="cl-404"></a>        if qfi_block_list[0, row_index, col_index, row_block_offset, col_block_offset] == qfi_nodata:
<a name="cl-405"></a>            recharge_block[row_index, col_index, row_block_offset, col_block_offset] = 0.0
<a name="cl-406"></a>            recharge_avail_block[row_index, col_index, row_block_offset, col_block_offset] = 0.0
<a name="cl-407"></a>            r_sum_avail_block[row_index, col_index, row_block_offset, col_block_offset] = 0.0
<a name="cl-408"></a>            cache_dirty[row_index, col_index] = 1
<a name="cl-409"></a>            continue
<a name="cl-410"></a>
<a name="cl-411"></a>        for direction_index in xrange(current_neighbor_index, 8):
<a name="cl-412"></a>            #get percent flow from neighbor to current cell
<a name="cl-413"></a>            neighbor_row = global_row + row_offsets[direction_index]
<a name="cl-414"></a>            neighbor_col = global_col + col_offsets[direction_index]
<a name="cl-415"></a>
<a name="cl-416"></a>            #See if neighbor out of bounds
<a name="cl-417"></a>            if (neighbor_row &lt; 0 or neighbor_row &gt;= n_rows or neighbor_col &lt; 0 or neighbor_col &gt;= n_cols):
<a name="cl-418"></a>                continue
<a name="cl-419"></a>
<a name="cl-420"></a>            block_cache.update_cache(neighbor_row, neighbor_col, &amp;neighbor_row_index, &amp;neighbor_col_index, &amp;neighbor_row_block_offset, &amp;neighbor_col_block_offset)
<a name="cl-421"></a>            #if neighbor inflows
<a name="cl-422"></a>            neighbor_direction = outflow_direction_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset]
<a name="cl-423"></a>            if neighbor_direction == outflow_direction_nodata:
<a name="cl-424"></a>                continue
<a name="cl-425"></a>
<a name="cl-426"></a>            #check if the cell flows directly, or is one index off
<a name="cl-427"></a>            if (inflow_offsets[direction_index] != neighbor_direction and
<a name="cl-428"></a>                    ((inflow_offsets[direction_index] - 1) % 8) != neighbor_direction):
<a name="cl-429"></a>                #then neighbor doesn't inflow into current cell
<a name="cl-430"></a>                continue
<a name="cl-431"></a>
<a name="cl-432"></a>            #Calculate the outflow weight
<a name="cl-433"></a>            outflow_weight = outflow_weights_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset]
<a name="cl-434"></a>
<a name="cl-435"></a>            if ((inflow_offsets[direction_index] - 1) % 8) == neighbor_direction:
<a name="cl-436"></a>                outflow_weight = 1.0 - outflow_weight
<a name="cl-437"></a>
<a name="cl-438"></a>            if outflow_weight &lt;= 0.0:
<a name="cl-439"></a>                continue
<a name="cl-440"></a>
<a name="cl-441"></a>            if r_sum_avail_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset] == recharge_nodata:
<a name="cl-442"></a>                #push current cell and and loop
<a name="cl-443"></a>                cells_to_process.push(current_index)
<a name="cl-444"></a>                cell_neighbor_to_process.push(direction_index)
<a name="cl-445"></a>                r_sum_stack.push(current_r_sum_avail)
<a name="cl-446"></a>                cells_to_process.push(neighbor_row * n_cols + neighbor_col)
<a name="cl-447"></a>                cell_neighbor_to_process.push(0)
<a name="cl-448"></a>                r_sum_stack.push(0.0)
<a name="cl-449"></a>                neighbors_calculated = 0
<a name="cl-450"></a>                break
<a name="cl-451"></a>            else:
<a name="cl-452"></a>                #'calculate r_avail_i and r_i'
<a name="cl-453"></a>                #add the contribution of the upstream to r_avail and r_i
<a name="cl-454"></a>                current_r_sum_avail += (
<a name="cl-455"></a>                    r_sum_avail_block[neighbor_row_index, neighbor_col_index,
<a name="cl-456"></a>                        neighbor_row_block_offset, neighbor_col_block_offset] +
<a name="cl-457"></a>                    recharge_avail_block[neighbor_row_index, neighbor_col_index,
<a name="cl-458"></a>                        neighbor_row_block_offset, neighbor_col_block_offset]) * outflow_weight
<a name="cl-459"></a>
<a name="cl-460"></a>        if not neighbors_calculated:
<a name="cl-461"></a>            continue
<a name="cl-462"></a>
<a name="cl-463"></a>        #if we got here current_r_sum_avail is correct
<a name="cl-464"></a>        block_cache.update_cache(global_row, global_col, &amp;row_index, &amp;col_index, &amp;row_block_offset, &amp;col_block_offset)
<a name="cl-465"></a>        p_i = 0.0
<a name="cl-466"></a>        qf_i = 0.0
<a name="cl-467"></a>        aet_sum = 0.0
<a name="cl-468"></a>        for month_index in xrange(N_MONTHS):
<a name="cl-469"></a>            p_m = precip_block_list[month_index, row_index, col_index, row_block_offset, col_block_offset]
<a name="cl-470"></a>            p_i += p_m
<a name="cl-471"></a>            pet_m = (
<a name="cl-472"></a>                kc_block[row_index, col_index, row_block_offset, col_block_offset] *
<a name="cl-473"></a>                et0_block_list[month_index, row_index, col_index, row_block_offset, col_block_offset])
<a name="cl-474"></a>            qfi_m = qfi_block_list[month_index, row_index, col_index, row_block_offset, col_block_offset]
<a name="cl-475"></a>            qf_i += qfi_m
<a name="cl-476"></a>            aet_m = min(
<a name="cl-477"></a>                pet_m, p_m - qfi_m + alpha_m * beta_i * current_r_sum_avail)
<a name="cl-478"></a>            aet_sum += aet_m
<a name="cl-479"></a>        r_i = p_i - qf_i - aet_sum
<a name="cl-480"></a>
<a name="cl-481"></a>        #if it's a stream, set recharge to 0 and ae to nodata
<a name="cl-482"></a>        if stream_block[row_index, col_index, row_block_offset, col_block_offset] == 1:
<a name="cl-483"></a>            r_i = 0
<a name="cl-484"></a>            aet_sum = aet_nodata
<a name="cl-485"></a>
<a name="cl-486"></a>        r_sum_avail_block[row_index, col_index, row_block_offset, col_block_offset] = current_r_sum_avail
<a name="cl-487"></a>        recharge_avail_block[row_index, col_index, row_block_offset, col_block_offset] = max(gamma*r_i, 0)
<a name="cl-488"></a>        recharge_block[row_index, col_index, row_block_offset, col_block_offset] = r_i
<a name="cl-489"></a>        aet_block[row_index, col_index, row_block_offset, col_block_offset] = aet_sum
<a name="cl-490"></a>        cache_dirty[row_index, col_index] = 1
<a name="cl-491"></a>
<a name="cl-492"></a>    block_cache.flush_cache()
<a name="cl-493"></a>
<a name="cl-494"></a>
<a name="cl-495"></a>#@cython.boundscheck(False)
<a name="cl-496"></a>@cython.wraparound(False)
<a name="cl-497"></a>@cython.cdivision(True)
<a name="cl-498"></a>def calculate_flow_weights(
<a name="cl-499"></a>    flow_direction_uri, outflow_weights_uri, outflow_direction_uri):
<a name="cl-500"></a>    """This function calculates the flow weights from a d-infinity based
<a name="cl-501"></a>        flow algorithm to assist in walking up the flow graph.
<a name="cl-502"></a>
<a name="cl-503"></a>        flow_direction_uri - uri to a flow direction GDAL dataset that's
<a name="cl-504"></a>            used to calculate the flow graph
<a name="cl-505"></a>        outflow_weights_uri - a uri to a float32 dataset that will be created
<a name="cl-506"></a>            whose elements correspond to the percent outflow from the current
<a name="cl-507"></a>            cell to its first counter-clockwise neighbor
<a name="cl-508"></a>        outflow_direction_uri - a uri to a byte dataset that will indicate the
<a name="cl-509"></a>            first counter clockwise outflow neighbor as an index from the
<a name="cl-510"></a>            following diagram
<a name="cl-511"></a>
<a name="cl-512"></a>            3 2 1
<a name="cl-513"></a>            4 x 0
<a name="cl-514"></a>            5 6 7
<a name="cl-515"></a>
<a name="cl-516"></a>        returns nothing"""
<a name="cl-517"></a>
<a name="cl-518"></a>    cdef time_t start
<a name="cl-519"></a>    time(&amp;start)
<a name="cl-520"></a>
<a name="cl-521"></a>    flow_direction_dataset = gdal.Open(flow_direction_uri)
<a name="cl-522"></a>    cdef double flow_direction_nodata
<a name="cl-523"></a>    flow_direction_band = flow_direction_dataset.GetRasterBand(1)
<a name="cl-524"></a>    flow_direction_nodata = flow_direction_band.GetNoDataValue()
<a name="cl-525"></a>
<a name="cl-526"></a>    cdef int block_col_size, block_row_size
<a name="cl-527"></a>    block_col_size, block_row_size = flow_direction_band.GetBlockSize()
<a name="cl-528"></a>
<a name="cl-529"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] flow_direction_block = numpy.empty(
<a name="cl-530"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-531"></a>
<a name="cl-532"></a>    #This is the array that's used to keep track of the connections of the
<a name="cl-533"></a>    #current cell to those *inflowing* to the cell, thus the 8 directions
<a name="cl-534"></a>    cdef int n_cols, n_rows
<a name="cl-535"></a>    n_cols, n_rows = flow_direction_band.XSize, flow_direction_band.YSize
<a name="cl-536"></a>
<a name="cl-537"></a>    cdef int outflow_direction_nodata = 9
<a name="cl-538"></a>    pygeoprocessing.new_raster_from_base_uri(
<a name="cl-539"></a>        flow_direction_uri, outflow_direction_uri, 'GTiff',
<a name="cl-540"></a>        outflow_direction_nodata, gdal.GDT_Byte, fill_value=outflow_direction_nodata)
<a name="cl-541"></a>    outflow_direction_dataset = gdal.Open(outflow_direction_uri, gdal.GA_Update)
<a name="cl-542"></a>    outflow_direction_band = outflow_direction_dataset.GetRasterBand(1)
<a name="cl-543"></a>    cdef numpy.ndarray[numpy.npy_byte, ndim=4] outflow_direction_block = (
<a name="cl-544"></a>        numpy.empty((N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.int8))
<a name="cl-545"></a>
<a name="cl-546"></a>    cdef double outflow_weights_nodata = -1.0
<a name="cl-547"></a>    pygeoprocessing.new_raster_from_base_uri(
<a name="cl-548"></a>        flow_direction_uri, outflow_weights_uri, 'GTiff',
<a name="cl-549"></a>        outflow_weights_nodata, gdal.GDT_Float32, fill_value=outflow_weights_nodata)
<a name="cl-550"></a>    outflow_weights_dataset = gdal.Open(outflow_weights_uri, gdal.GA_Update)
<a name="cl-551"></a>    outflow_weights_band = outflow_weights_dataset.GetRasterBand(1)
<a name="cl-552"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] outflow_weights_block = (
<a name="cl-553"></a>        numpy.empty((N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32))
<a name="cl-554"></a>
<a name="cl-555"></a>    #center point of global index
<a name="cl-556"></a>    cdef int global_row, global_col, global_block_row, global_block_col #index into the overall raster
<a name="cl-557"></a>    cdef int row_index, col_index #the index of the cache block
<a name="cl-558"></a>    cdef int row_block_offset, col_block_offset #index into the cache block
<a name="cl-559"></a>
<a name="cl-560"></a>    #neighbor sections of global index
<a name="cl-561"></a>    cdef int neighbor_row, neighbor_col #neighbor equivalent of global_{row,col}
<a name="cl-562"></a>    cdef int neighbor_row_index, neighbor_col_index #neighbor cache index
<a name="cl-563"></a>    cdef int neighbor_row_block_offset, neighbor_col_block_offset #index into the neighbor cache block
<a name="cl-564"></a>
<a name="cl-565"></a>    #define all the caches
<a name="cl-566"></a>    cdef numpy.ndarray[numpy.npy_int8, ndim=2] cache_dirty = numpy.zeros(
<a name="cl-567"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.int8)
<a name="cl-568"></a>
<a name="cl-569"></a>    cache_dirty[:] = 0
<a name="cl-570"></a>    band_list = [flow_direction_band, outflow_direction_band, outflow_weights_band]
<a name="cl-571"></a>    block_list = [flow_direction_block, outflow_direction_block, outflow_weights_block]
<a name="cl-572"></a>    update_list = [False, True, True]
<a name="cl-573"></a>
<a name="cl-574"></a>    cdef BlockCache_SWY block_cache = BlockCache_SWY(
<a name="cl-575"></a>        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size, block_col_size, band_list, block_list, update_list, cache_dirty)
<a name="cl-576"></a>
<a name="cl-577"></a>
<a name="cl-578"></a>    #The number of diagonal offsets defines the neighbors, angle between them
<a name="cl-579"></a>    #and the actual angle to point to the neighbor
<a name="cl-580"></a>    cdef int n_neighbors = 8
<a name="cl-581"></a>    cdef double angle_to_neighbor[8]
<a name="cl-582"></a>    for index in range(8):
<a name="cl-583"></a>        angle_to_neighbor[index] = 2.0*PI*index/8.0
<a name="cl-584"></a>
<a name="cl-585"></a>    #diagonal offsets index is 0, 1, 2, 3, 4, 5, 6, 7 from the figure above
<a name="cl-586"></a>    cdef int *diagonal_offsets = [
<a name="cl-587"></a>        1, -n_cols+1, -n_cols, -n_cols-1, -1, n_cols-1, n_cols, n_cols+1]
<a name="cl-588"></a>
<a name="cl-589"></a>    #Iterate over flow directions
<a name="cl-590"></a>    cdef int neighbor_direction_index
<a name="cl-591"></a>    cdef long current_index
<a name="cl-592"></a>    cdef double flow_direction, flow_angle_to_neighbor, outflow_weight
<a name="cl-593"></a>
<a name="cl-594"></a>    cdef time_t last_time, current_time
<a name="cl-595"></a>    time(&amp;last_time)
<a name="cl-596"></a>    for global_block_row in xrange(int(ceil(float(n_rows) / block_row_size))):
<a name="cl-597"></a>        time(&amp;current_time)
<a name="cl-598"></a>        if current_time - last_time &gt; 5.0:
<a name="cl-599"></a>            LOGGER.info("calculate_flow_weights %.1f%% complete", (global_row + 1.0) / n_rows * 100)
<a name="cl-600"></a>            last_time = current_time
<a name="cl-601"></a>        for global_block_col in xrange(int(ceil(float(n_cols) / block_col_size))):
<a name="cl-602"></a>            for global_row in xrange(global_block_row*block_row_size, min((global_block_row+1)*block_row_size, n_rows)):
<a name="cl-603"></a>                for global_col in xrange(global_block_col*block_col_size, min((global_block_col+1)*block_col_size, n_cols)):
<a name="cl-604"></a>                    block_cache.update_cache(global_row, global_col, &amp;row_index, &amp;col_index, &amp;row_block_offset, &amp;col_block_offset)
<a name="cl-605"></a>                    flow_direction = flow_direction_block[row_index, col_index, row_block_offset, col_block_offset]
<a name="cl-606"></a>                    #make sure the flow direction is defined, if not, skip this cell
<a name="cl-607"></a>                    if flow_direction == flow_direction_nodata:
<a name="cl-608"></a>                        continue
<a name="cl-609"></a>                    found = False
<a name="cl-610"></a>                    for neighbor_direction_index in range(n_neighbors):
<a name="cl-611"></a>                        flow_angle_to_neighbor = abs(angle_to_neighbor[neighbor_direction_index] - flow_direction)
<a name="cl-612"></a>                        if flow_angle_to_neighbor &lt;= PI/4.0:
<a name="cl-613"></a>                            found = True
<a name="cl-614"></a>
<a name="cl-615"></a>                            #Determine if the direction we're on is oriented at 90
<a name="cl-616"></a>                            #degrees or 45 degrees.  Given our orientation even number
<a name="cl-617"></a>                            #neighbor indexes are oriented 90 degrees and odd are 45
<a name="cl-618"></a>                            outflow_weight = 0.0
<a name="cl-619"></a>
<a name="cl-620"></a>                            if neighbor_direction_index % 2 == 0:
<a name="cl-621"></a>                                outflow_weight = 1.0 - tan(flow_angle_to_neighbor)
<a name="cl-622"></a>                            else:
<a name="cl-623"></a>                                outflow_weight = tan(PI/4.0 - flow_angle_to_neighbor)
<a name="cl-624"></a>
<a name="cl-625"></a>                            # clamping the outflow weight in case it's too large or small
<a name="cl-626"></a>                            if outflow_weight &gt;= 1.0 - 1e-6:
<a name="cl-627"></a>                                outflow_weight = 1.0
<a name="cl-628"></a>                            if outflow_weight &lt;= 1e-6:
<a name="cl-629"></a>                                outflow_weight = 1.0
<a name="cl-630"></a>                                neighbor_direction_index = (neighbor_direction_index + 1) % 8
<a name="cl-631"></a>                            outflow_direction_block[row_index, col_index, row_block_offset, col_block_offset] = neighbor_direction_index
<a name="cl-632"></a>                            outflow_weights_block[row_index, col_index, row_block_offset, col_block_offset] = outflow_weight
<a name="cl-633"></a>                            cache_dirty[row_index, col_index] = 1
<a name="cl-634"></a>
<a name="cl-635"></a>                            #we found the outflow direction
<a name="cl-636"></a>                            break
<a name="cl-637"></a>                    if not found:
<a name="cl-638"></a>                        LOGGER.warn('no flow direction found for %s %s' % \
<a name="cl-639"></a>                                         (row_index, col_index))
<a name="cl-640"></a>    block_cache.flush_cache()
<a name="cl-641"></a>
<a name="cl-642"></a>cdef struct Row_Col_Weight_Tuple:
<a name="cl-643"></a>    int row_index
<a name="cl-644"></a>    int col_index
<a name="cl-645"></a>    int weight
<a name="cl-646"></a>
<a name="cl-647"></a>
<a name="cl-648"></a>def fill_pits(dem_uri, dem_out_uri):
<a name="cl-649"></a>    """This function fills regions in a DEM that don't drain to the edge
<a name="cl-650"></a>        of the dataset.  The resulting DEM will likely have plateaus where the
<a name="cl-651"></a>        pits are filled.
<a name="cl-652"></a>
<a name="cl-653"></a>        dem_uri - the original dem URI
<a name="cl-654"></a>        dem_out_uri - the original dem with pits raised to the highest drain
<a name="cl-655"></a>            value
<a name="cl-656"></a>
<a name="cl-657"></a>        returns nothing"""
<a name="cl-658"></a>
<a name="cl-659"></a>    cdef int *row_offsets = [0, -1, -1, -1,  0,  1, 1, 1]
<a name="cl-660"></a>    cdef int *col_offsets = [1,  1,  0, -1, -1, -1, 0, 1]
<a name="cl-661"></a>
<a name="cl-662"></a>    dem_ds = gdal.Open(dem_uri, gdal.GA_ReadOnly)
<a name="cl-663"></a>    cdef int n_rows = dem_ds.RasterYSize
<a name="cl-664"></a>    cdef int n_cols = dem_ds.RasterXSize
<a name="cl-665"></a>
<a name="cl-666"></a>    dem_band = dem_ds.GetRasterBand(1)
<a name="cl-667"></a>
<a name="cl-668"></a>    #copy the dem to a different dataset so we know the type
<a name="cl-669"></a>    dem_band = dem_ds.GetRasterBand(1)
<a name="cl-670"></a>    raw_nodata_value = pygeoprocessing.get_nodata_from_uri(dem_uri)
<a name="cl-671"></a>
<a name="cl-672"></a>    cdef double nodata_value
<a name="cl-673"></a>    if raw_nodata_value is not None:
<a name="cl-674"></a>        nodata_value = raw_nodata_value
<a name="cl-675"></a>    else:
<a name="cl-676"></a>        LOGGER.warn("Nodata value not set, defaulting to -9999.9")
<a name="cl-677"></a>        nodata_value = -9999.9
<a name="cl-678"></a>    pygeoprocessing.new_raster_from_base_uri(
<a name="cl-679"></a>        dem_uri, dem_out_uri, 'GTiff', nodata_value, gdal.GDT_Float32,
<a name="cl-680"></a>        INF)
<a name="cl-681"></a>    dem_out_ds = gdal.Open(dem_out_uri, gdal.GA_Update)
<a name="cl-682"></a>    dem_out_band = dem_out_ds.GetRasterBand(1)
<a name="cl-683"></a>    cdef int row_index, col_index, neighbor_index
<a name="cl-684"></a>    cdef float min_dem_value, cur_dem_value, neighbor_dem_value
<a name="cl-685"></a>    cdef int pit_count = 0
<a name="cl-686"></a>
<a name="cl-687"></a>    for row_index in range(n_rows):
<a name="cl-688"></a>        dem_out_array = dem_band.ReadAsArray(
<a name="cl-689"></a>            xoff=0, yoff=row_index, win_xsize=n_cols, win_ysize=1)
<a name="cl-690"></a>        dem_out_band.WriteArray(dem_out_array, xoff=0, yoff=row_index)
<a name="cl-691"></a>
<a name="cl-692"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=2] dem_array
<a name="cl-693"></a>
<a name="cl-694"></a>    for row_index in range(1, n_rows - 1):
<a name="cl-695"></a>        #load 3 rows at a time
<a name="cl-696"></a>        dem_array = dem_out_band.ReadAsArray(
<a name="cl-697"></a>            xoff=0, yoff=row_index-1, win_xsize=n_cols, win_ysize=3)
<a name="cl-698"></a>
<a name="cl-699"></a>        for col_index in range(1, n_cols - 1):
<a name="cl-700"></a>            min_dem_value = nodata_value
<a name="cl-701"></a>            cur_dem_value = dem_array[1, col_index]
<a name="cl-702"></a>            if cur_dem_value == nodata_value:
<a name="cl-703"></a>                continue
<a name="cl-704"></a>            for neighbor_index in range(8):
<a name="cl-705"></a>                neighbor_dem_value = dem_array[
<a name="cl-706"></a>                    1 + row_offsets[neighbor_index],
<a name="cl-707"></a>                    col_index + col_offsets[neighbor_index]]
<a name="cl-708"></a>                if neighbor_dem_value == nodata_value:
<a name="cl-709"></a>                    continue
<a name="cl-710"></a>                if (neighbor_dem_value &lt; min_dem_value or
<a name="cl-711"></a>                    min_dem_value == nodata_value):
<a name="cl-712"></a>                    min_dem_value = neighbor_dem_value
<a name="cl-713"></a>            if min_dem_value &gt; cur_dem_value:
<a name="cl-714"></a>                #it's a pit, bump it up
<a name="cl-715"></a>                dem_array[1, col_index] = min_dem_value
<a name="cl-716"></a>                pit_count += 1
<a name="cl-717"></a>
<a name="cl-718"></a>        dem_out_band.WriteArray(
<a name="cl-719"></a>            dem_array[1, :].reshape((1,n_cols)), xoff=0, yoff=row_index)
<a name="cl-720"></a>
<a name="cl-721"></a>
<a name="cl-722"></a>#@cython.boundscheck(False)
<a name="cl-723"></a>@cython.wraparound(False)
<a name="cl-724"></a>@cython.cdivision(True)
<a name="cl-725"></a>def flow_direction_inf(dem_uri, flow_direction_uri):
<a name="cl-726"></a>    """Calculates the D-infinity flow algorithm.  The output is a float
<a name="cl-727"></a>        raster whose values range from 0 to 2pi.
<a name="cl-728"></a>
<a name="cl-729"></a>        Algorithm from: Tarboton, "A new method for the determination of flow
<a name="cl-730"></a>        directions and upslope areas in grid digital elevation models," Water
<a name="cl-731"></a>        Resources Research, vol. 33, no. 2, pages 309 - 319, February 1997.
<a name="cl-732"></a>
<a name="cl-733"></a>        Also resolves flow directions in flat areas of DEM.
<a name="cl-734"></a>
<a name="cl-735"></a>        dem_uri (string) - (input) a uri to a single band GDAL Dataset with elevation values
<a name="cl-736"></a>        flow_direction_uri - (input/output) a uri to an existing GDAL dataset with
<a name="cl-737"></a>            of same as dem_uri.  Flow direction will be defined in regions that have
<a name="cl-738"></a>            nodata values in them.  non-nodata values will be ignored.  This is so
<a name="cl-739"></a>            this function can be used as a two pass filter for resolving flow directions
<a name="cl-740"></a>            on a raw dem, then filling plateaus and doing another pass.
<a name="cl-741"></a>
<a name="cl-742"></a>       returns nothing"""
<a name="cl-743"></a>
<a name="cl-744"></a>    cdef int col_index, row_index, n_cols, n_rows, max_index, facet_index, flat_index
<a name="cl-745"></a>    cdef double e_0, e_1, e_2, s_1, s_2, d_1, d_2, flow_direction, slope, \
<a name="cl-746"></a>        flow_direction_max_slope, slope_max, nodata_flow
<a name="cl-747"></a>
<a name="cl-748"></a>    cdef double dem_nodata = pygeoprocessing.get_nodata_from_uri(dem_uri)
<a name="cl-749"></a>    #if it is not set, set it to a traditional nodata value
<a name="cl-750"></a>    if dem_nodata == None:
<a name="cl-751"></a>        dem_nodata = -9999
<a name="cl-752"></a>
<a name="cl-753"></a>    dem_ds = gdal.Open(dem_uri)
<a name="cl-754"></a>    dem_band = dem_ds.GetRasterBand(1)
<a name="cl-755"></a>
<a name="cl-756"></a>    #facet elevation and factors for slope and flow_direction calculations
<a name="cl-757"></a>    #from Table 1 in Tarboton 1997.
<a name="cl-758"></a>    #THIS IS IMPORTANT:  The order is row (j), column (i), transposed to GDAL
<a name="cl-759"></a>    #convention.
<a name="cl-760"></a>    cdef int *e_0_offsets = [+0, +0,
<a name="cl-761"></a>                             +0, +0,
<a name="cl-762"></a>                             +0, +0,
<a name="cl-763"></a>                             +0, +0,
<a name="cl-764"></a>                             +0, +0,
<a name="cl-765"></a>                             +0, +0,
<a name="cl-766"></a>                             +0, +0,
<a name="cl-767"></a>                             +0, +0]
<a name="cl-768"></a>    cdef int *e_1_offsets = [+0, +1,
<a name="cl-769"></a>                             -1, +0,
<a name="cl-770"></a>                             -1, +0,
<a name="cl-771"></a>                             +0, -1,
<a name="cl-772"></a>                             +0, -1,
<a name="cl-773"></a>                             +1, +0,
<a name="cl-774"></a>                             +1, +0,
<a name="cl-775"></a>                             +0, +1]
<a name="cl-776"></a>    cdef int *e_2_offsets = [-1, +1,
<a name="cl-777"></a>                             -1, +1,
<a name="cl-778"></a>                             -1, -1,
<a name="cl-779"></a>                             -1, -1,
<a name="cl-780"></a>                             +1, -1,
<a name="cl-781"></a>                             +1, -1,
<a name="cl-782"></a>                             +1, +1,
<a name="cl-783"></a>                             +1, +1]
<a name="cl-784"></a>    cdef int *a_c = [0, 1, 1, 2, 2, 3, 3, 4]
<a name="cl-785"></a>    cdef int *a_f = [1, -1, 1, -1, 1, -1, 1, -1]
<a name="cl-786"></a>
<a name="cl-787"></a>    cdef int *row_offsets = [0, -1, -1, -1,  0,  1, 1, 1]
<a name="cl-788"></a>    cdef int *col_offsets = [1,  1,  0, -1, -1, -1, 0, 1]
<a name="cl-789"></a>
<a name="cl-790"></a>    n_rows, n_cols = pygeoprocessing.get_row_col_from_uri(dem_uri)
<a name="cl-791"></a>    d_1 = pygeoprocessing.get_cell_size_from_uri(dem_uri)
<a name="cl-792"></a>    d_2 = d_1
<a name="cl-793"></a>    cdef double max_r = numpy.pi / 4.0
<a name="cl-794"></a>
<a name="cl-795"></a>    #Create a flow carray and respective dataset
<a name="cl-796"></a>    cdef float flow_nodata = -9999
<a name="cl-797"></a>    pygeoprocessing.new_raster_from_base_uri(
<a name="cl-798"></a>        dem_uri, flow_direction_uri, 'GTiff', flow_nodata,
<a name="cl-799"></a>        gdal.GDT_Float32, fill_value=flow_nodata)
<a name="cl-800"></a>
<a name="cl-801"></a>    flow_direction_dataset = gdal.Open(flow_direction_uri, gdal.GA_Update)
<a name="cl-802"></a>    flow_band = flow_direction_dataset.GetRasterBand(1)
<a name="cl-803"></a>
<a name="cl-804"></a>    #center point of global index
<a name="cl-805"></a>    cdef int block_row_size, block_col_size
<a name="cl-806"></a>    block_col_size, block_row_size = dem_band.GetBlockSize()
<a name="cl-807"></a>    cdef int global_row, global_col, e_0_row, e_0_col, e_1_row, e_1_col, e_2_row, e_2_col #index into the overall raster
<a name="cl-808"></a>    cdef int e_0_row_index, e_0_col_index #the index of the cache block
<a name="cl-809"></a>    cdef int e_0_row_block_offset, e_0_col_block_offset #index into the cache block
<a name="cl-810"></a>    cdef int e_1_row_index, e_1_col_index #the index of the cache block
<a name="cl-811"></a>    cdef int e_1_row_block_offset, e_1_col_block_offset #index into the cache block
<a name="cl-812"></a>    cdef int e_2_row_index, e_2_col_index #the index of the cache block
<a name="cl-813"></a>    cdef int e_2_row_block_offset, e_2_col_block_offset #index into the cache block
<a name="cl-814"></a>
<a name="cl-815"></a>    cdef int global_block_row, global_block_col #used to walk the global blocks
<a name="cl-816"></a>
<a name="cl-817"></a>    #neighbor sections of global index
<a name="cl-818"></a>    cdef int neighbor_row, neighbor_col #neighbor equivalent of global_{row,col}
<a name="cl-819"></a>    cdef int neighbor_row_index, neighbor_col_index #neighbor cache index
<a name="cl-820"></a>    cdef int neighbor_row_block_offset, neighbor_col_block_offset #index into the neighbor cache block
<a name="cl-821"></a>
<a name="cl-822"></a>    #define all the caches
<a name="cl-823"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] flow_block = numpy.zeros(
<a name="cl-824"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-825"></a>    #DEM block is a 64 bit float so it can capture the resolution of small DEM offsets
<a name="cl-826"></a>    #from the plateau resolution algorithm.
<a name="cl-827"></a>    cdef numpy.ndarray[numpy.npy_float64, ndim=4] dem_block = numpy.zeros(
<a name="cl-828"></a>      (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float64)
<a name="cl-829"></a>
<a name="cl-830"></a>    #the BlockCache_SWY object needs parallel lists of bands, blocks, and boolean tags to indicate which ones are updated
<a name="cl-831"></a>    band_list = [dem_band, flow_band]
<a name="cl-832"></a>    block_list = [dem_block, flow_block]
<a name="cl-833"></a>    update_list = [False, True]
<a name="cl-834"></a>    cdef numpy.ndarray[numpy.npy_byte, ndim=2] cache_dirty = numpy.zeros((N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.byte)
<a name="cl-835"></a>
<a name="cl-836"></a>    cdef BlockCache_SWY block_cache = BlockCache_SWY(
<a name="cl-837"></a>        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size, block_col_size, band_list, block_list, update_list, cache_dirty)
<a name="cl-838"></a>
<a name="cl-839"></a>    cdef int row_offset, col_offset
<a name="cl-840"></a>
<a name="cl-841"></a>    cdef int n_global_block_rows = int(ceil(float(n_rows) / block_row_size))
<a name="cl-842"></a>    cdef int n_global_block_cols = int(ceil(float(n_cols) / block_col_size))
<a name="cl-843"></a>    cdef time_t last_time, current_time
<a name="cl-844"></a>    cdef float current_flow
<a name="cl-845"></a>    time(&amp;last_time)
<a name="cl-846"></a>    #flow not defined on the edges, so just go 1 row in
<a name="cl-847"></a>    for global_block_row in xrange(n_global_block_rows):
<a name="cl-848"></a>        time(&amp;current_time)
<a name="cl-849"></a>        if current_time - last_time &gt; 5.0:
<a name="cl-850"></a>            LOGGER.info("flow_direction_inf %.1f%% complete", (global_row + 1.0) / n_rows * 100)
<a name="cl-851"></a>            last_time = current_time
<a name="cl-852"></a>        for global_block_col in xrange(n_global_block_cols):
<a name="cl-853"></a>            for global_row in xrange(global_block_row*block_row_size, min((global_block_row+1)*block_row_size, n_rows)):
<a name="cl-854"></a>                for global_col in xrange(global_block_col*block_col_size, min((global_block_col+1)*block_col_size, n_cols)):
<a name="cl-855"></a>                    #is cache block not loaded?
<a name="cl-856"></a>
<a name="cl-857"></a>                    e_0_row = e_0_offsets[0] + global_row
<a name="cl-858"></a>                    e_0_col = e_0_offsets[1] + global_col
<a name="cl-859"></a>
<a name="cl-860"></a>                    block_cache.update_cache(e_0_row, e_0_col, &amp;e_0_row_index, &amp;e_0_col_index, &amp;e_0_row_block_offset, &amp;e_0_col_block_offset)
<a name="cl-861"></a>
<a name="cl-862"></a>                    e_0 = dem_block[e_0_row_index, e_0_col_index, e_0_row_block_offset, e_0_col_block_offset]
<a name="cl-863"></a>                    #skip if we're on a nodata pixel skip
<a name="cl-864"></a>                    if e_0 == dem_nodata:
<a name="cl-865"></a>                        continue
<a name="cl-866"></a>
<a name="cl-867"></a>                    #Calculate the flow flow_direction for each facet
<a name="cl-868"></a>                    slope_max = 0 #use this to keep track of the maximum down-slope
<a name="cl-869"></a>                    flow_direction_max_slope = 0 #flow direction on max downward slope
<a name="cl-870"></a>                    max_index = 0 #index to keep track of max slope facet
<a name="cl-871"></a>
<a name="cl-872"></a>                    for facet_index in range(8):
<a name="cl-873"></a>                        #This defines the three points the facet
<a name="cl-874"></a>
<a name="cl-875"></a>                        e_1_row = e_1_offsets[facet_index * 2 + 0] + global_row
<a name="cl-876"></a>                        e_1_col = e_1_offsets[facet_index * 2 + 1] + global_col
<a name="cl-877"></a>                        e_2_row = e_2_offsets[facet_index * 2 + 0] + global_row
<a name="cl-878"></a>                        e_2_col = e_2_offsets[facet_index * 2 + 1] + global_col
<a name="cl-879"></a>                        #make sure one of the facets doesn't hang off the edge
<a name="cl-880"></a>                        if (e_1_row &lt; 0 or e_1_row &gt;= n_rows or
<a name="cl-881"></a>                            e_2_row &lt; 0 or e_2_row &gt;= n_rows or
<a name="cl-882"></a>                            e_1_col &lt; 0 or e_1_col &gt;= n_cols or
<a name="cl-883"></a>                            e_2_col &lt; 0 or e_2_col &gt;= n_cols):
<a name="cl-884"></a>                            continue
<a name="cl-885"></a>
<a name="cl-886"></a>                        block_cache.update_cache(e_1_row, e_1_col, &amp;e_1_row_index, &amp;e_1_col_index, &amp;e_1_row_block_offset, &amp;e_1_col_block_offset)
<a name="cl-887"></a>                        block_cache.update_cache(e_2_row, e_2_col, &amp;e_2_row_index, &amp;e_2_col_index, &amp;e_2_row_block_offset, &amp;e_2_col_block_offset)
<a name="cl-888"></a>
<a name="cl-889"></a>                        e_1 = dem_block[e_1_row_index, e_1_col_index, e_1_row_block_offset, e_1_col_block_offset]
<a name="cl-890"></a>                        e_2 = dem_block[e_2_row_index, e_2_col_index, e_2_row_block_offset, e_2_col_block_offset]
<a name="cl-891"></a>
<a name="cl-892"></a>                        if e_1 == dem_nodata and e_2 == dem_nodata:
<a name="cl-893"></a>                            continue
<a name="cl-894"></a>
<a name="cl-895"></a>                        #s_1 is slope along straight edge
<a name="cl-896"></a>                        s_1 = (e_0 - e_1) / d_1 #Eqn 1
<a name="cl-897"></a>                        #slope along diagonal edge
<a name="cl-898"></a>                        s_2 = (e_1 - e_2) / d_2 #Eqn 2
<a name="cl-899"></a>
<a name="cl-900"></a>                        #can't calculate flow direction if one of the facets is nodata
<a name="cl-901"></a>                        if e_1 == dem_nodata or e_2 == dem_nodata:
<a name="cl-902"></a>                            #calc max slope here
<a name="cl-903"></a>                            if e_1 != dem_nodata and facet_index % 2 == 0 and e_1 &lt; e_0:
<a name="cl-904"></a>                                #straight line to next pixel
<a name="cl-905"></a>                                slope = s_1
<a name="cl-906"></a>                                flow_direction = 0
<a name="cl-907"></a>                            elif e_2 != dem_nodata and facet_index % 2 == 1 and e_2 &lt; e_0:
<a name="cl-908"></a>                                #diagonal line to next pixel
<a name="cl-909"></a>                                slope = (e_0 - e_2) / sqrt(d_1 **2 + d_2 ** 2)
<a name="cl-910"></a>                                flow_direction = max_r
<a name="cl-911"></a>                            else:
<a name="cl-912"></a>                                continue
<a name="cl-913"></a>                        else:
<a name="cl-914"></a>                            #both facets are defined, this is the core of
<a name="cl-915"></a>                            #d-infinity algorithm
<a name="cl-916"></a>                            flow_direction = atan2(s_2, s_1) #Eqn 3
<a name="cl-917"></a>
<a name="cl-918"></a>                            if flow_direction &lt; 0: #Eqn 4
<a name="cl-919"></a>                                #If the flow direction goes off one side, set flow
<a name="cl-920"></a>                                #direction to that side and the slope to the straight line
<a name="cl-921"></a>                                #distance slope
<a name="cl-922"></a>                                flow_direction = 0
<a name="cl-923"></a>                                slope = s_1
<a name="cl-924"></a>                            elif flow_direction &gt; max_r: #Eqn 5
<a name="cl-925"></a>                                #If the flow direciton goes off the diagonal side, figure
<a name="cl-926"></a>                                #out what its value is and
<a name="cl-927"></a>                                flow_direction = max_r
<a name="cl-928"></a>                                slope = (e_0 - e_2) / sqrt(d_1 ** 2 + d_2 ** 2)
<a name="cl-929"></a>                            else:
<a name="cl-930"></a>                                slope = sqrt(s_1 ** 2 + s_2 ** 2) #Eqn 3
<a name="cl-931"></a>
<a name="cl-932"></a>                        #update the maxes depending on the results above
<a name="cl-933"></a>                        if slope &gt; slope_max:
<a name="cl-934"></a>                            flow_direction_max_slope = flow_direction
<a name="cl-935"></a>                            slope_max = slope
<a name="cl-936"></a>                            max_index = facet_index
<a name="cl-937"></a>
<a name="cl-938"></a>                    #if there's a downward slope, save the flow direction
<a name="cl-939"></a>                    if slope_max &gt; 0:
<a name="cl-940"></a>                        flow_block[e_0_row_index, e_0_col_index, e_0_row_block_offset, e_0_col_block_offset] = (
<a name="cl-941"></a>                            a_f[max_index] * flow_direction_max_slope +
<a name="cl-942"></a>                            a_c[max_index] * PI / 2.0)
<a name="cl-943"></a>                        cache_dirty[e_0_row_index, e_0_col_index] = 1
<a name="cl-944"></a>
<a name="cl-945"></a>    block_cache.flush_cache()
<a name="cl-946"></a>    flow_band = None
<a name="cl-947"></a>    gdal.Dataset.__swig_destroy__(flow_direction_dataset)
<a name="cl-948"></a>    flow_direction_dataset = None
<a name="cl-949"></a>    pygeoprocessing.calculate_raster_stats_uri(flow_direction_uri)
<a name="cl-950"></a>
<a name="cl-951"></a>
<a name="cl-952"></a>#@cython.boundscheck(False)
<a name="cl-953"></a>@cython.wraparound(False)
<a name="cl-954"></a>@cython.cdivision(True)
<a name="cl-955"></a>def distance_to_stream(
<a name="cl-956"></a>        flow_direction_uri, stream_uri, distance_uri, factor_uri=None):
<a name="cl-957"></a>    """This function calculates the flow downhill distance to the stream layers
<a name="cl-958"></a>
<a name="cl-959"></a>        Args:
<a name="cl-960"></a>            flow_direction_uri (string) - (input) a path to a raster with
<a name="cl-961"></a>                d-infinity flow directions.
<a name="cl-962"></a>            stream_uri (string) - (input) a raster where 1 indicates a stream
<a name="cl-963"></a>                all other values ignored must be same dimensions and projection
<a name="cl-964"></a>                as flow_direction_uri.
<a name="cl-965"></a>            distance_uri (string) - (output) a path to the output raster that
<a name="cl-966"></a>                will be created as same dimensions as the input rasters where
<a name="cl-967"></a>                each pixel is in linear units the drainage from that point to a
<a name="cl-968"></a>                stream.
<a name="cl-969"></a>            factor_uri (string) - (optional input) a floating point raster that
<a name="cl-970"></a>                is used to multiply the stepsize by for each current pixel,
<a name="cl-971"></a>                useful for some models to calculate a user defined downstream
<a name="cl-972"></a>                factor.
<a name="cl-973"></a>
<a name="cl-974"></a>        Returns:
<a name="cl-975"></a>            nothing"""
<a name="cl-976"></a>
<a name="cl-977"></a>    cdef float distance_nodata = -9999
<a name="cl-978"></a>    pygeoprocessing.new_raster_from_base_uri(
<a name="cl-979"></a>        flow_direction_uri, distance_uri, 'GTiff', distance_nodata,
<a name="cl-980"></a>        gdal.GDT_Float32, fill_value=distance_nodata)
<a name="cl-981"></a>
<a name="cl-982"></a>    cdef float processed_cell_nodata = 127
<a name="cl-983"></a>    processed_cell_uri = (
<a name="cl-984"></a>        os.path.join(os.path.dirname(flow_direction_uri), 'processed_cell.tif'))
<a name="cl-985"></a>    pygeoprocessing.new_raster_from_base_uri(
<a name="cl-986"></a>        distance_uri, processed_cell_uri, 'GTiff', processed_cell_nodata,
<a name="cl-987"></a>        gdal.GDT_Byte, fill_value=0)
<a name="cl-988"></a>
<a name="cl-989"></a>    processed_cell_ds = gdal.Open(processed_cell_uri, gdal.GA_Update)
<a name="cl-990"></a>    processed_cell_band = processed_cell_ds.GetRasterBand(1)
<a name="cl-991"></a>
<a name="cl-992"></a>    cdef int *row_offsets = [0, -1, -1, -1,  0,  1, 1, 1]
<a name="cl-993"></a>    cdef int *col_offsets = [1,  1,  0, -1, -1, -1, 0, 1]
<a name="cl-994"></a>    cdef int *inflow_offsets = [4, 5, 6, 7, 0, 1, 2, 3]
<a name="cl-995"></a>
<a name="cl-996"></a>    cdef int n_rows, n_cols
<a name="cl-997"></a>    n_rows, n_cols = pygeoprocessing.get_row_col_from_uri(
<a name="cl-998"></a>        flow_direction_uri)
<a name="cl-999"></a>    cdef int INF = n_rows + n_cols
<a name="cl-1000"></a>
<a name="cl-1001"></a>    cdef deque[int] visit_stack
<a name="cl-1002"></a>
<a name="cl-1003"></a>    stream_ds = gdal.Open(stream_uri)
<a name="cl-1004"></a>    stream_band = stream_ds.GetRasterBand(1)
<a name="cl-1005"></a>    cdef float stream_nodata = pygeoprocessing.get_nodata_from_uri(
<a name="cl-1006"></a>        stream_uri)
<a name="cl-1007"></a>    cdef float cell_size = pygeoprocessing.get_cell_size_from_uri(stream_uri)
<a name="cl-1008"></a>
<a name="cl-1009"></a>    distance_ds = gdal.Open(distance_uri, gdal.GA_Update)
<a name="cl-1010"></a>    distance_band = distance_ds.GetRasterBand(1)
<a name="cl-1011"></a>
<a name="cl-1012"></a>    outflow_weights_uri = pygeoprocessing.temporary_filename()
<a name="cl-1013"></a>    outflow_direction_uri = pygeoprocessing.temporary_filename()
<a name="cl-1014"></a>    calculate_flow_weights(
<a name="cl-1015"></a>        flow_direction_uri, outflow_weights_uri, outflow_direction_uri)
<a name="cl-1016"></a>    outflow_weights_ds = gdal.Open(outflow_weights_uri)
<a name="cl-1017"></a>    outflow_weights_band = outflow_weights_ds.GetRasterBand(1)
<a name="cl-1018"></a>    cdef float outflow_weights_nodata = pygeoprocessing.get_nodata_from_uri(
<a name="cl-1019"></a>        outflow_weights_uri)
<a name="cl-1020"></a>    outflow_direction_ds = gdal.Open(outflow_direction_uri)
<a name="cl-1021"></a>    outflow_direction_band = outflow_direction_ds.GetRasterBand(1)
<a name="cl-1022"></a>    cdef int outflow_direction_nodata = pygeoprocessing.get_nodata_from_uri(
<a name="cl-1023"></a>        outflow_direction_uri)
<a name="cl-1024"></a>    cdef int block_col_size, block_row_size
<a name="cl-1025"></a>    block_col_size, block_row_size = stream_band.GetBlockSize()
<a name="cl-1026"></a>    cdef int n_global_block_rows = int(ceil(float(n_rows) / block_row_size))
<a name="cl-1027"></a>    cdef int n_global_block_cols = int(ceil(float(n_cols) / block_col_size))
<a name="cl-1028"></a>
<a name="cl-1029"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] stream_block = numpy.zeros(
<a name="cl-1030"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
<a name="cl-1031"></a>        dtype=numpy.float32)
<a name="cl-1032"></a>    cdef numpy.ndarray[numpy.npy_int8, ndim=4] outflow_direction_block = (
<a name="cl-1033"></a>        numpy.zeros(
<a name="cl-1034"></a>            (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
<a name="cl-1035"></a>            dtype=numpy.int8))
<a name="cl-1036"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] outflow_weights_block = (
<a name="cl-1037"></a>        numpy.zeros(
<a name="cl-1038"></a>            (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
<a name="cl-1039"></a>            dtype=numpy.float32))
<a name="cl-1040"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] distance_block = numpy.zeros(
<a name="cl-1041"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
<a name="cl-1042"></a>        dtype=numpy.float32)
<a name="cl-1043"></a>    cdef numpy.ndarray[numpy.npy_int8, ndim=4] processed_cell_block = (
<a name="cl-1044"></a>        numpy.zeros(
<a name="cl-1045"></a>            (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
<a name="cl-1046"></a>            dtype=numpy.int8))
<a name="cl-1047"></a>
<a name="cl-1048"></a>    band_list = [stream_band, outflow_direction_band, outflow_weights_band,
<a name="cl-1049"></a>                 distance_band, processed_cell_band]
<a name="cl-1050"></a>    block_list = [stream_block, outflow_direction_block, outflow_weights_block,
<a name="cl-1051"></a>                  distance_block, processed_cell_block]
<a name="cl-1052"></a>    update_list = [False, False, False, True, True]
<a name="cl-1053"></a>
<a name="cl-1054"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] factor_block
<a name="cl-1055"></a>    cdef int factor_exists = (factor_uri != None)
<a name="cl-1056"></a>    if factor_exists:
<a name="cl-1057"></a>        factor_block = numpy.zeros(
<a name="cl-1058"></a>            (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
<a name="cl-1059"></a>            dtype=numpy.float32)
<a name="cl-1060"></a>        factor_ds = gdal.Open(factor_uri)
<a name="cl-1061"></a>        factor_band = factor_ds.GetRasterBand(1)
<a name="cl-1062"></a>        band_list.append(factor_band)
<a name="cl-1063"></a>        block_list.append(factor_block)
<a name="cl-1064"></a>        update_list.append(False)
<a name="cl-1065"></a>
<a name="cl-1066"></a>    cdef numpy.ndarray[numpy.npy_byte, ndim=2] cache_dirty = (
<a name="cl-1067"></a>        numpy.zeros((N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.byte))
<a name="cl-1068"></a>
<a name="cl-1069"></a>    cdef BlockCache_SWY block_cache = BlockCache_SWY(
<a name="cl-1070"></a>        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size,
<a name="cl-1071"></a>        block_col_size, band_list, block_list, update_list, cache_dirty)
<a name="cl-1072"></a>
<a name="cl-1073"></a>    #center point of global index
<a name="cl-1074"></a>    cdef int global_row, global_col
<a name="cl-1075"></a>    cdef int row_index, col_index
<a name="cl-1076"></a>    cdef int row_block_offset, col_block_offset
<a name="cl-1077"></a>    cdef int global_block_row, global_block_col
<a name="cl-1078"></a>
<a name="cl-1079"></a>    #neighbor sections of global index
<a name="cl-1080"></a>    cdef int neighbor_row, neighbor_col
<a name="cl-1081"></a>    cdef int neighbor_row_index, neighbor_col_index
<a name="cl-1082"></a>    cdef int neighbor_row_block_offset, neighbor_col_block_offset
<a name="cl-1083"></a>    cdef int flat_index
<a name="cl-1084"></a>
<a name="cl-1085"></a>    cdef float original_distance
<a name="cl-1086"></a>
<a name="cl-1087"></a>    cdef c_set[int] cells_in_queue
<a name="cl-1088"></a>
<a name="cl-1089"></a>    #build up the stream pixel indexes as starting seed points for the search
<a name="cl-1090"></a>    cdef time_t last_time, current_time
<a name="cl-1091"></a>    time(&amp;last_time)
<a name="cl-1092"></a>    for global_block_row in xrange(n_global_block_rows):
<a name="cl-1093"></a>        time(&amp;current_time)
<a name="cl-1094"></a>        if current_time - last_time &gt; 5.0:
<a name="cl-1095"></a>            LOGGER.info(
<a name="cl-1096"></a>                "find_sinks %.1f%% complete",
<a name="cl-1097"></a>                (global_block_row + 1.0) / n_global_block_rows * 100)
<a name="cl-1098"></a>            last_time = current_time
<a name="cl-1099"></a>        for global_block_col in xrange(n_global_block_cols):
<a name="cl-1100"></a>            for global_row in xrange(
<a name="cl-1101"></a>                    global_block_row*block_row_size,
<a name="cl-1102"></a>                    min((global_block_row+1)*block_row_size, n_rows)):
<a name="cl-1103"></a>                for global_col in xrange(
<a name="cl-1104"></a>                        global_block_col*block_col_size,
<a name="cl-1105"></a>                        min((global_block_col+1)*block_col_size, n_cols)):
<a name="cl-1106"></a>                    block_cache.update_cache(
<a name="cl-1107"></a>                        global_row, global_col, &amp;row_index, &amp;col_index,
<a name="cl-1108"></a>                        &amp;row_block_offset, &amp;col_block_offset)
<a name="cl-1109"></a>                    if stream_block[
<a name="cl-1110"></a>                            row_index, col_index, row_block_offset,
<a name="cl-1111"></a>                            col_block_offset] == 1:
<a name="cl-1112"></a>                        flat_index = global_row * n_cols + global_col
<a name="cl-1113"></a>                        visit_stack.push_front(global_row * n_cols + global_col)
<a name="cl-1114"></a>                        cells_in_queue.insert(flat_index)
<a name="cl-1115"></a>
<a name="cl-1116"></a>                        distance_block[row_index, col_index,
<a name="cl-1117"></a>                            row_block_offset, col_block_offset] = 0
<a name="cl-1118"></a>                        processed_cell_block[row_index, col_index,
<a name="cl-1119"></a>                            row_block_offset, col_block_offset] = 1
<a name="cl-1120"></a>                        cache_dirty[row_index, col_index] = 1
<a name="cl-1121"></a>
<a name="cl-1122"></a>    cdef int neighbor_outflow_direction, neighbor_index, outflow_direction
<a name="cl-1123"></a>    cdef float neighbor_outflow_weight, current_distance, cell_travel_distance
<a name="cl-1124"></a>    cdef float outflow_weight, neighbor_distance, step_size
<a name="cl-1125"></a>    cdef float factor
<a name="cl-1126"></a>    cdef int it_flows_here
<a name="cl-1127"></a>    cdef int downstream_index, downstream_calculated
<a name="cl-1128"></a>    cdef float downstream_distance
<a name="cl-1129"></a>    cdef float current_stream
<a name="cl-1130"></a>    cdef int pushed_current = False
<a name="cl-1131"></a>
<a name="cl-1132"></a>    while visit_stack.size() &gt; 0:
<a name="cl-1133"></a>        flat_index = visit_stack.front()
<a name="cl-1134"></a>        visit_stack.pop_front()
<a name="cl-1135"></a>        cells_in_queue.erase(flat_index)
<a name="cl-1136"></a>        global_row = flat_index / n_cols
<a name="cl-1137"></a>        global_col = flat_index % n_cols
<a name="cl-1138"></a>
<a name="cl-1139"></a>        block_cache.update_cache(
<a name="cl-1140"></a>            global_row, global_col, &amp;row_index, &amp;col_index,
<a name="cl-1141"></a>            &amp;row_block_offset, &amp;col_block_offset)
<a name="cl-1142"></a>
<a name="cl-1143"></a>        update_downstream = False
<a name="cl-1144"></a>        current_distance = 0.0
<a name="cl-1145"></a>
<a name="cl-1146"></a>        time(&amp;current_time)
<a name="cl-1147"></a>        if current_time - last_time &gt; 5.0:
<a name="cl-1148"></a>            last_time = current_time
<a name="cl-1149"></a>            LOGGER.info(
<a name="cl-1150"></a>                'visit_stack on stream distance size: %d ', visit_stack.size())
<a name="cl-1151"></a>
<a name="cl-1152"></a>        current_stream = stream_block[
<a name="cl-1153"></a>            row_index, col_index, row_block_offset, col_block_offset]
<a name="cl-1154"></a>        outflow_direction = outflow_direction_block[
<a name="cl-1155"></a>            row_index, col_index, row_block_offset,
<a name="cl-1156"></a>            col_block_offset]
<a name="cl-1157"></a>        if current_stream == 1:
<a name="cl-1158"></a>            distance_block[row_index, col_index,
<a name="cl-1159"></a>                row_block_offset, col_block_offset] = 0
<a name="cl-1160"></a>            processed_cell_block[row_index, col_index,
<a name="cl-1161"></a>                row_block_offset, col_block_offset] = 1
<a name="cl-1162"></a>            cache_dirty[row_index, col_index] = 1
<a name="cl-1163"></a>        elif outflow_direction == outflow_direction_nodata:
<a name="cl-1164"></a>            current_distance = INF
<a name="cl-1165"></a>        elif processed_cell_block[row_index, col_index, row_block_offset,
<a name="cl-1166"></a>                col_block_offset] == 0:
<a name="cl-1167"></a>            #add downstream distance to current distance
<a name="cl-1168"></a>
<a name="cl-1169"></a>            outflow_weight = outflow_weights_block[
<a name="cl-1170"></a>                row_index, col_index, row_block_offset,
<a name="cl-1171"></a>                col_block_offset]
<a name="cl-1172"></a>
<a name="cl-1173"></a>            if factor_exists:
<a name="cl-1174"></a>                factor = factor_block[
<a name="cl-1175"></a>                    row_index, col_index, row_block_offset, col_block_offset]
<a name="cl-1176"></a>            else:
<a name="cl-1177"></a>                factor = 1.0
<a name="cl-1178"></a>
<a name="cl-1179"></a>            for neighbor_index in xrange(2):
<a name="cl-1180"></a>                #check if downstream neighbors are calcualted
<a name="cl-1181"></a>                if neighbor_index == 1:
<a name="cl-1182"></a>                    outflow_direction = (outflow_direction + 1) % 8
<a name="cl-1183"></a>                    outflow_weight = (1.0 - outflow_weight)
<a name="cl-1184"></a>
<a name="cl-1185"></a>                if outflow_weight &lt;= 0.0:
<a name="cl-1186"></a>                    continue
<a name="cl-1187"></a>
<a name="cl-1188"></a>                neighbor_row = global_row + row_offsets[outflow_direction]
<a name="cl-1189"></a>                neighbor_col = global_col + col_offsets[outflow_direction]
<a name="cl-1190"></a>                if (neighbor_row &lt; 0 or neighbor_row &gt;= n_rows or
<a name="cl-1191"></a>                        neighbor_col &lt; 0 or neighbor_col &gt;= n_cols):
<a name="cl-1192"></a>                    #out of bounds
<a name="cl-1193"></a>                    continue
<a name="cl-1194"></a>
<a name="cl-1195"></a>                block_cache.update_cache(
<a name="cl-1196"></a>                    neighbor_row, neighbor_col, &amp;neighbor_row_index,
<a name="cl-1197"></a>                    &amp;neighbor_col_index, &amp;neighbor_row_block_offset,
<a name="cl-1198"></a>                    &amp;neighbor_col_block_offset)
<a name="cl-1199"></a>
<a name="cl-1200"></a>                if stream_block[neighbor_row_index,
<a name="cl-1201"></a>                        neighbor_col_index, neighbor_row_block_offset,
<a name="cl-1202"></a>                        neighbor_col_block_offset] == stream_nodata:
<a name="cl-1203"></a>                    #out of the valid raster entirely
<a name="cl-1204"></a>                    continue
<a name="cl-1205"></a>
<a name="cl-1206"></a>                neighbor_distance = distance_block[
<a name="cl-1207"></a>                    neighbor_row_index, neighbor_col_index,
<a name="cl-1208"></a>                    neighbor_row_block_offset, neighbor_col_block_offset]
<a name="cl-1209"></a>
<a name="cl-1210"></a>                neighbor_outflow_direction = outflow_direction_block[
<a name="cl-1211"></a>                    neighbor_row_index, neighbor_col_index,
<a name="cl-1212"></a>                    neighbor_row_block_offset, neighbor_col_block_offset]
<a name="cl-1213"></a>
<a name="cl-1214"></a>                neighbor_outflow_weight = outflow_weights_block[
<a name="cl-1215"></a>                    neighbor_row_index, neighbor_col_index,
<a name="cl-1216"></a>                    neighbor_row_block_offset, neighbor_col_block_offset]
<a name="cl-1217"></a>
<a name="cl-1218"></a>                if processed_cell_block[neighbor_row_index, neighbor_col_index,
<a name="cl-1219"></a>                        neighbor_row_block_offset,
<a name="cl-1220"></a>                        neighbor_col_block_offset] == 0:
<a name="cl-1221"></a>                    neighbor_flat_index = neighbor_row * n_cols + neighbor_col
<a name="cl-1222"></a>                    #insert into the processing queue if it's not already there
<a name="cl-1223"></a>                    if (cells_in_queue.find(flat_index) ==
<a name="cl-1224"></a>                            cells_in_queue.end()):
<a name="cl-1225"></a>                        visit_stack.push_back(flat_index)
<a name="cl-1226"></a>                        cells_in_queue.insert(flat_index)
<a name="cl-1227"></a>
<a name="cl-1228"></a>                    if (cells_in_queue.find(neighbor_flat_index) ==
<a name="cl-1229"></a>                            cells_in_queue.end()):
<a name="cl-1230"></a>                        visit_stack.push_front(neighbor_flat_index)
<a name="cl-1231"></a>                        cells_in_queue.insert(neighbor_flat_index)
<a name="cl-1232"></a>
<a name="cl-1233"></a>                    update_downstream = True
<a name="cl-1234"></a>                    neighbor_distance = 0.0
<a name="cl-1235"></a>
<a name="cl-1236"></a>                if outflow_direction % 2 == 1:
<a name="cl-1237"></a>                    #increase distance by a square root of 2 for diagonal
<a name="cl-1238"></a>                    step_size = cell_size * 1.41421356237
<a name="cl-1239"></a>                else:
<a name="cl-1240"></a>                    step_size = cell_size
<a name="cl-1241"></a>
<a name="cl-1242"></a>                current_distance += (
<a name="cl-1243"></a>                    neighbor_distance + step_size * factor) * outflow_weight
<a name="cl-1244"></a>
<a name="cl-1245"></a>        if not update_downstream:
<a name="cl-1246"></a>            #mark flat_index as processed
<a name="cl-1247"></a>            block_cache.update_cache(
<a name="cl-1248"></a>                global_row, global_col, &amp;row_index, &amp;col_index,
<a name="cl-1249"></a>                &amp;row_block_offset, &amp;col_block_offset)
<a name="cl-1250"></a>            processed_cell_block[row_index, col_index,
<a name="cl-1251"></a>                row_block_offset, col_block_offset] = 1
<a name="cl-1252"></a>            distance_block[row_index, col_index,
<a name="cl-1253"></a>                row_block_offset, col_block_offset] = current_distance
<a name="cl-1254"></a>            cache_dirty[row_index, col_index] = 1
<a name="cl-1255"></a>
<a name="cl-1256"></a>            #update any upstream neighbors with this distance
<a name="cl-1257"></a>            for neighbor_index in range(8):
<a name="cl-1258"></a>                neighbor_row = global_row + row_offsets[neighbor_index]
<a name="cl-1259"></a>                neighbor_col = global_col + col_offsets[neighbor_index]
<a name="cl-1260"></a>                if (neighbor_row &lt; 0 or neighbor_row &gt;= n_rows or
<a name="cl-1261"></a>                        neighbor_col &lt; 0 or neighbor_col &gt;= n_cols):
<a name="cl-1262"></a>                    #out of bounds
<a name="cl-1263"></a>                    continue
<a name="cl-1264"></a>
<a name="cl-1265"></a>                block_cache.update_cache(
<a name="cl-1266"></a>                    neighbor_row, neighbor_col, &amp;neighbor_row_index,
<a name="cl-1267"></a>                    &amp;neighbor_col_index, &amp;neighbor_row_block_offset,
<a name="cl-1268"></a>                    &amp;neighbor_col_block_offset)
<a name="cl-1269"></a>
<a name="cl-1270"></a>                #streams were already added, skip if they are in the queue
<a name="cl-1271"></a>                if (stream_block[neighbor_row_index, neighbor_col_index,
<a name="cl-1272"></a>                        neighbor_row_block_offset,
<a name="cl-1273"></a>                        neighbor_col_block_offset] == 1 or
<a name="cl-1274"></a>                    stream_block[neighbor_row_index, neighbor_col_index,
<a name="cl-1275"></a>                        neighbor_row_block_offset,
<a name="cl-1276"></a>                        neighbor_col_block_offset] == stream_nodata):
<a name="cl-1277"></a>                    continue
<a name="cl-1278"></a>
<a name="cl-1279"></a>                if processed_cell_block[
<a name="cl-1280"></a>                        neighbor_row_index,
<a name="cl-1281"></a>                        neighbor_col_index,
<a name="cl-1282"></a>                        neighbor_row_block_offset,
<a name="cl-1283"></a>                        neighbor_col_block_offset] == 1:
<a name="cl-1284"></a>                    #don't reprocess it, it's already been updated by two valid
<a name="cl-1285"></a>                    #children
<a name="cl-1286"></a>                    continue
<a name="cl-1287"></a>
<a name="cl-1288"></a>                neighbor_outflow_direction = outflow_direction_block[
<a name="cl-1289"></a>                    neighbor_row_index, neighbor_col_index,
<a name="cl-1290"></a>                    neighbor_row_block_offset, neighbor_col_block_offset]
<a name="cl-1291"></a>                if neighbor_outflow_direction == outflow_direction_nodata:
<a name="cl-1292"></a>                    #if the neighbor has no flow, we can't flow here
<a name="cl-1293"></a>                    continue
<a name="cl-1294"></a>
<a name="cl-1295"></a>                neighbor_outflow_weight = outflow_weights_block[
<a name="cl-1296"></a>                    neighbor_row_index, neighbor_col_index,
<a name="cl-1297"></a>                    neighbor_row_block_offset, neighbor_col_block_offset]
<a name="cl-1298"></a>
<a name="cl-1299"></a>                it_flows_here = False
<a name="cl-1300"></a>                if (neighbor_outflow_direction ==
<a name="cl-1301"></a>                        inflow_offsets[neighbor_index]):
<a name="cl-1302"></a>                    it_flows_here = True
<a name="cl-1303"></a>                elif ((neighbor_outflow_direction + 1) % 8 ==
<a name="cl-1304"></a>                        inflow_offsets[neighbor_index]):
<a name="cl-1305"></a>                    it_flows_here = True
<a name="cl-1306"></a>                    neighbor_outflow_weight = 1.0 - neighbor_outflow_weight
<a name="cl-1307"></a>
<a name="cl-1308"></a>                neighbor_flat_index = neighbor_row * n_cols + neighbor_col
<a name="cl-1309"></a>                if (it_flows_here and neighbor_outflow_weight &gt; 0.0 and
<a name="cl-1310"></a>                    cells_in_queue.find(neighbor_flat_index) ==
<a name="cl-1311"></a>                        cells_in_queue.end()):
<a name="cl-1312"></a>                    visit_stack.push_back(neighbor_flat_index)
<a name="cl-1313"></a>                    cells_in_queue.insert(neighbor_flat_index)
<a name="cl-1314"></a>
<a name="cl-1315"></a>    block_cache.flush_cache()
<a name="cl-1316"></a>
<a name="cl-1317"></a>    for dataset in [outflow_weights_ds, outflow_direction_ds]:
<a name="cl-1318"></a>        gdal.Dataset.__swig_destroy__(dataset)
<a name="cl-1319"></a>    for dataset_uri in [outflow_weights_uri, outflow_direction_uri]:
<a name="cl-1320"></a>        os.remove(dataset_uri)
<a name="cl-1321"></a>
<a name="cl-1322"></a>
<a name="cl-1323"></a>#@cython.boundscheck(False)
<a name="cl-1324"></a>@cython.wraparound(False)
<a name="cl-1325"></a>def percent_to_sink(
<a name="cl-1326"></a>    sink_pixels_uri, export_rate_uri, outflow_direction_uri,
<a name="cl-1327"></a>    outflow_weights_uri, effect_uri):
<a name="cl-1328"></a>    """This function calculates the amount of load from a single pixel
<a name="cl-1329"></a>        to the source pixels given the percent export rate per pixel.
<a name="cl-1330"></a>
<a name="cl-1331"></a>        sink_pixels_uri - the pixels of interest that will receive flux.
<a name="cl-1332"></a>            This may be a set of stream pixels, or a single pixel at a
<a name="cl-1333"></a>            watershed outlet.
<a name="cl-1334"></a>
<a name="cl-1335"></a>        export_rate_uri - a GDAL floating point dataset that has a percent
<a name="cl-1336"></a>            of flux exported per pixel
<a name="cl-1337"></a>
<a name="cl-1338"></a>        outflow_direction_uri - a uri to a byte dataset that indicates the
<a name="cl-1339"></a>            first counter clockwise outflow neighbor as an index from the
<a name="cl-1340"></a>            following diagram
<a name="cl-1341"></a>
<a name="cl-1342"></a>            3 2 1
<a name="cl-1343"></a>            4 x 0
<a name="cl-1344"></a>            5 6 7
<a name="cl-1345"></a>
<a name="cl-1346"></a>        outflow_weights_uri - a uri to a float32 dataset whose elements
<a name="cl-1347"></a>            correspond to the percent outflow from the current cell to its
<a name="cl-1348"></a>            first counter-clockwise neighbor
<a name="cl-1349"></a>
<a name="cl-1350"></a>        effect_uri - the output GDAL dataset that shows the percent of flux
<a name="cl-1351"></a>            emanating per pixel that will reach any sink pixel
<a name="cl-1352"></a>
<a name="cl-1353"></a>        returns nothing"""
<a name="cl-1354"></a>
<a name="cl-1355"></a>    LOGGER.info("calculating percent to sink")
<a name="cl-1356"></a>    cdef time_t start_time
<a name="cl-1357"></a>    time(&amp;start_time)
<a name="cl-1358"></a>
<a name="cl-1359"></a>    sink_pixels_dataset = gdal.Open(sink_pixels_uri)
<a name="cl-1360"></a>    sink_pixels_band = sink_pixels_dataset.GetRasterBand(1)
<a name="cl-1361"></a>    cdef int sink_pixels_nodata = pygeoprocessing.get_nodata_from_uri(
<a name="cl-1362"></a>        sink_pixels_uri)
<a name="cl-1363"></a>    export_rate_dataset = gdal.Open(export_rate_uri)
<a name="cl-1364"></a>    export_rate_band = export_rate_dataset.GetRasterBand(1)
<a name="cl-1365"></a>    cdef double export_rate_nodata = pygeoprocessing.get_nodata_from_uri(
<a name="cl-1366"></a>        export_rate_uri)
<a name="cl-1367"></a>    outflow_direction_dataset = gdal.Open(outflow_direction_uri)
<a name="cl-1368"></a>    outflow_direction_band = outflow_direction_dataset.GetRasterBand(1)
<a name="cl-1369"></a>    cdef int outflow_direction_nodata = pygeoprocessing.get_nodata_from_uri(
<a name="cl-1370"></a>        outflow_direction_uri)
<a name="cl-1371"></a>    outflow_weights_dataset = gdal.Open(outflow_weights_uri)
<a name="cl-1372"></a>    outflow_weights_band = outflow_weights_dataset.GetRasterBand(1)
<a name="cl-1373"></a>    cdef float outflow_weights_nodata = pygeoprocessing.get_nodata_from_uri(
<a name="cl-1374"></a>        outflow_weights_uri)
<a name="cl-1375"></a>
<a name="cl-1376"></a>    cdef int block_col_size, block_row_size
<a name="cl-1377"></a>    block_col_size, block_row_size = sink_pixels_band.GetBlockSize()
<a name="cl-1378"></a>    cdef int n_rows = sink_pixels_dataset.RasterYSize
<a name="cl-1379"></a>    cdef int n_cols = sink_pixels_dataset.RasterXSize
<a name="cl-1380"></a>
<a name="cl-1381"></a>    cdef double effect_nodata = -1.0
<a name="cl-1382"></a>    pygeoprocessing.new_raster_from_base_uri(
<a name="cl-1383"></a>        sink_pixels_uri, effect_uri, 'GTiff', effect_nodata,
<a name="cl-1384"></a>        gdal.GDT_Float32, fill_value=effect_nodata)
<a name="cl-1385"></a>    effect_dataset = gdal.Open(effect_uri, gdal.GA_Update)
<a name="cl-1386"></a>    effect_band = effect_dataset.GetRasterBand(1)
<a name="cl-1387"></a>
<a name="cl-1388"></a>    #center point of global index
<a name="cl-1389"></a>    cdef int global_row, global_col #index into the overall raster
<a name="cl-1390"></a>    cdef int row_index, col_index #the index of the cache block
<a name="cl-1391"></a>    cdef int row_block_offset, col_block_offset #index into the cache block
<a name="cl-1392"></a>    cdef int global_block_row, global_block_col #used to walk the global blocks
<a name="cl-1393"></a>
<a name="cl-1394"></a>    #neighbor sections of global index
<a name="cl-1395"></a>    cdef int neighbor_row, neighbor_col #neighbor equivalent of global_{row,col}
<a name="cl-1396"></a>    cdef int neighbor_row_index, neighbor_col_index #neighbor cache index
<a name="cl-1397"></a>    cdef int neighbor_row_block_offset, neighbor_col_block_offset #index into the neighbor cache block
<a name="cl-1398"></a>
<a name="cl-1399"></a>    #define all the caches
<a name="cl-1400"></a>
<a name="cl-1401"></a>    cdef numpy.ndarray[numpy.npy_int32, ndim=4] sink_pixels_block = numpy.zeros(
<a name="cl-1402"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.int32)
<a name="cl-1403"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] export_rate_block = numpy.zeros(
<a name="cl-1404"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-1405"></a>    cdef numpy.ndarray[numpy.npy_int8, ndim=4] outflow_direction_block = numpy.zeros(
<a name="cl-1406"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.int8)
<a name="cl-1407"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] outflow_weights_block = numpy.zeros(
<a name="cl-1408"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-1409"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] out_block = numpy.zeros(
<a name="cl-1410"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-1411"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] effect_block = numpy.zeros(
<a name="cl-1412"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-1413"></a>    #the BlockCache_SWY object needs parallel lists of bands, blocks, and boolean tags to indicate which ones are updated
<a name="cl-1414"></a>    block_list = [sink_pixels_block, export_rate_block, outflow_direction_block, outflow_weights_block, effect_block]
<a name="cl-1415"></a>    band_list = [sink_pixels_band, export_rate_band, outflow_direction_band, outflow_weights_band, effect_band]
<a name="cl-1416"></a>    update_list = [False, False, False, False, True]
<a name="cl-1417"></a>    cdef numpy.ndarray[numpy.npy_byte, ndim=2] cache_dirty = numpy.zeros((N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.byte)
<a name="cl-1418"></a>
<a name="cl-1419"></a>    cdef BlockCache_SWY block_cache = BlockCache_SWY(
<a name="cl-1420"></a>        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size, block_col_size, band_list, block_list, update_list, cache_dirty)
<a name="cl-1421"></a>
<a name="cl-1422"></a>    cdef float outflow_weight, neighbor_outflow_weight
<a name="cl-1423"></a>    cdef int neighbor_outflow_direction
<a name="cl-1424"></a>
<a name="cl-1425"></a>    #Diagonal offsets are based off the following index notation for neighbors
<a name="cl-1426"></a>    #    3 2 1
<a name="cl-1427"></a>    #    4 p 0
<a name="cl-1428"></a>    #    5 6 7
<a name="cl-1429"></a>
<a name="cl-1430"></a>    cdef int *row_offsets = [0, -1, -1, -1,  0,  1, 1, 1]
<a name="cl-1431"></a>    cdef int *col_offsets = [1,  1,  0, -1, -1, -1, 0, 1]
<a name="cl-1432"></a>    cdef int *inflow_offsets = [4, 5, 6, 7, 0, 1, 2, 3]
<a name="cl-1433"></a>    cdef int flat_index
<a name="cl-1434"></a>    cdef deque[int] process_queue
<a name="cl-1435"></a>    #Queue the sinks
<a name="cl-1436"></a>    for global_block_row in xrange(int(numpy.ceil(float(n_rows) / block_row_size))):
<a name="cl-1437"></a>        for global_block_col in xrange(int(numpy.ceil(float(n_cols) / block_col_size))):
<a name="cl-1438"></a>            for global_row in xrange(global_block_row*block_row_size, min((global_block_row+1)*block_row_size, n_rows)):
<a name="cl-1439"></a>                for global_col in xrange(global_block_col*block_col_size, min((global_block_col+1)*block_col_size, n_cols)):
<a name="cl-1440"></a>                    block_cache.update_cache(global_row, global_col, &amp;row_index, &amp;col_index, &amp;row_block_offset, &amp;col_block_offset)
<a name="cl-1441"></a>                    if sink_pixels_block[row_index, col_index, row_block_offset, col_block_offset] == 1:
<a name="cl-1442"></a>                        effect_block[row_index, col_index, row_block_offset, col_block_offset] = 1.0
<a name="cl-1443"></a>                        cache_dirty[row_index, col_index] = 1
<a name="cl-1444"></a>                        process_queue.push_back(global_row * n_cols + global_col)
<a name="cl-1445"></a>
<a name="cl-1446"></a>    while process_queue.size() &gt; 0:
<a name="cl-1447"></a>        flat_index = process_queue.front()
<a name="cl-1448"></a>        process_queue.pop_front()
<a name="cl-1449"></a>        with cython.cdivision(True):
<a name="cl-1450"></a>            global_row = flat_index / n_cols
<a name="cl-1451"></a>            global_col = flat_index % n_cols
<a name="cl-1452"></a>
<a name="cl-1453"></a>        block_cache.update_cache(global_row, global_col, &amp;row_index, &amp;col_index, &amp;row_block_offset, &amp;col_block_offset)
<a name="cl-1454"></a>        if export_rate_block[row_index, col_index, row_block_offset, col_block_offset] == export_rate_nodata:
<a name="cl-1455"></a>            continue
<a name="cl-1456"></a>
<a name="cl-1457"></a>        #if the outflow weight is nodata, then not a valid pixel
<a name="cl-1458"></a>        outflow_weight = outflow_weights_block[row_index, col_index, row_block_offset, col_block_offset]
<a name="cl-1459"></a>        if outflow_weight == outflow_weights_nodata:
<a name="cl-1460"></a>            continue
<a name="cl-1461"></a>
<a name="cl-1462"></a>        for neighbor_index in range(8):
<a name="cl-1463"></a>            neighbor_row = global_row + row_offsets[neighbor_index]
<a name="cl-1464"></a>            neighbor_col = global_col + col_offsets[neighbor_index]
<a name="cl-1465"></a>            if neighbor_row &lt; 0 or neighbor_row &gt;= n_rows or neighbor_col &lt; 0 or neighbor_col &gt;= n_cols:
<a name="cl-1466"></a>                #out of bounds
<a name="cl-1467"></a>                continue
<a name="cl-1468"></a>
<a name="cl-1469"></a>            block_cache.update_cache(neighbor_row, neighbor_col, &amp;neighbor_row_index, &amp;neighbor_col_index, &amp;neighbor_row_block_offset, &amp;neighbor_col_block_offset)
<a name="cl-1470"></a>
<a name="cl-1471"></a>            if sink_pixels_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset] == 1:
<a name="cl-1472"></a>                #it's already a sink
<a name="cl-1473"></a>                continue
<a name="cl-1474"></a>
<a name="cl-1475"></a>            neighbor_outflow_direction = (
<a name="cl-1476"></a>                outflow_direction_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset])
<a name="cl-1477"></a>            #if the neighbor is no data, don't try to set that
<a name="cl-1478"></a>            if neighbor_outflow_direction == outflow_direction_nodata:
<a name="cl-1479"></a>                continue
<a name="cl-1480"></a>
<a name="cl-1481"></a>            neighbor_outflow_weight = (
<a name="cl-1482"></a>                outflow_weights_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset])
<a name="cl-1483"></a>            #if the neighbor is no data, don't try to set that
<a name="cl-1484"></a>            if neighbor_outflow_weight == outflow_direction_nodata:
<a name="cl-1485"></a>                continue
<a name="cl-1486"></a>
<a name="cl-1487"></a>            it_flows_here = False
<a name="cl-1488"></a>            if neighbor_outflow_direction == inflow_offsets[neighbor_index]:
<a name="cl-1489"></a>                #the neighbor flows into this cell
<a name="cl-1490"></a>                it_flows_here = True
<a name="cl-1491"></a>
<a name="cl-1492"></a>            if (neighbor_outflow_direction - 1) % 8 == inflow_offsets[neighbor_index]:
<a name="cl-1493"></a>                #the offset neighbor flows into this cell
<a name="cl-1494"></a>                it_flows_here = True
<a name="cl-1495"></a>                neighbor_outflow_weight = 1.0 - neighbor_outflow_weight
<a name="cl-1496"></a>
<a name="cl-1497"></a>            if it_flows_here:
<a name="cl-1498"></a>                #If we haven't processed that effect yet, set it to 0 and append to the queue
<a name="cl-1499"></a>                if effect_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset] == effect_nodata:
<a name="cl-1500"></a>                    process_queue.push_back(neighbor_row * n_cols + neighbor_col)
<a name="cl-1501"></a>                    effect_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset] = 0.0
<a name="cl-1502"></a>                    cache_dirty[neighbor_row_index, neighbor_col_index] = 1
<a name="cl-1503"></a>
<a name="cl-1504"></a>                #the percent of the pixel upstream equals the current percent
<a name="cl-1505"></a>                #times the percent flow to that pixels times the
<a name="cl-1506"></a>                effect_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset] += (
<a name="cl-1507"></a>                    effect_block[row_index, col_index, row_block_offset, col_block_offset] *
<a name="cl-1508"></a>                    neighbor_outflow_weight *
<a name="cl-1509"></a>                    export_rate_block[row_index, col_index, row_block_offset, col_block_offset])
<a name="cl-1510"></a>                cache_dirty[neighbor_row_index, neighbor_col_index] = 1
<a name="cl-1511"></a>
<a name="cl-1512"></a>    block_cache.flush_cache()
<a name="cl-1513"></a>    cdef time_t end_time
<a name="cl-1514"></a>    time(&amp;end_time)
<a name="cl-1515"></a>    LOGGER.info('Done calculating percent to sink elapsed time %ss' % \
<a name="cl-1516"></a>                    (end_time - start_time))
<a name="cl-1517"></a>
<a name="cl-1518"></a>
<a name="cl-1519"></a>#@cython.boundscheck(False)
<a name="cl-1520"></a>@cython.wraparound(False)
<a name="cl-1521"></a>@cython.cdivision(True)
<a name="cl-1522"></a>cdef flat_edges(
<a name="cl-1523"></a>        dem_uri, flow_direction_uri, deque[int] &amp;high_edges,
<a name="cl-1524"></a>        deque[int] &amp;low_edges, int drain_off_edge=0):
<a name="cl-1525"></a>    """This function locates flat cells that border on higher and lower terrain
<a name="cl-1526"></a>        and places them into sets for further processing.
<a name="cl-1527"></a>
<a name="cl-1528"></a>        Args:
<a name="cl-1529"></a>
<a name="cl-1530"></a>            dem_uri (string) - (input) a uri to a single band GDAL Dataset with
<a name="cl-1531"></a>                elevation values
<a name="cl-1532"></a>            flow_direction_uri (string) - (input/output) a uri to a single band
<a name="cl-1533"></a>                GDAL Dataset with partially defined d_infinity flow directions
<a name="cl-1534"></a>            high_edges (deque) - (output) will contain all the high edge cells as
<a name="cl-1535"></a>                flat row major order indexes
<a name="cl-1536"></a>            low_edges (deque) - (output) will contain all the low edge cells as flat
<a name="cl-1537"></a>                row major order indexes
<a name="cl-1538"></a>            drain_off_edge (int) - (input) if True will drain flat regions off
<a name="cl-1539"></a>                the nodata edge of a DEM"""
<a name="cl-1540"></a>
<a name="cl-1541"></a>    high_edges.clear()
<a name="cl-1542"></a>    low_edges.clear()
<a name="cl-1543"></a>
<a name="cl-1544"></a>    cdef int *neighbor_row_offset = [0, -1, -1, -1,  0,  1, 1, 1]
<a name="cl-1545"></a>    cdef int *neighbor_col_offset = [1,  1,  0, -1, -1, -1, 0, 1]
<a name="cl-1546"></a>
<a name="cl-1547"></a>    dem_ds = gdal.Open(dem_uri)
<a name="cl-1548"></a>    dem_band = dem_ds.GetRasterBand(1)
<a name="cl-1549"></a>    flow_ds = gdal.Open(flow_direction_uri, gdal.GA_Update)
<a name="cl-1550"></a>    flow_band = flow_ds.GetRasterBand(1)
<a name="cl-1551"></a>
<a name="cl-1552"></a>    cdef int block_col_size, block_row_size
<a name="cl-1553"></a>    block_col_size, block_row_size = dem_band.GetBlockSize()
<a name="cl-1554"></a>    cdef int n_rows = dem_ds.RasterYSize
<a name="cl-1555"></a>    cdef int n_cols = dem_ds.RasterXSize
<a name="cl-1556"></a>
<a name="cl-1557"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] flow_block = numpy.zeros(
<a name="cl-1558"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
<a name="cl-1559"></a>        dtype=numpy.float32)
<a name="cl-1560"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] dem_block = numpy.zeros(
<a name="cl-1561"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
<a name="cl-1562"></a>        dtype=numpy.float32)
<a name="cl-1563"></a>
<a name="cl-1564"></a>    band_list = [dem_band, flow_band]
<a name="cl-1565"></a>    block_list = [dem_block, flow_block]
<a name="cl-1566"></a>    update_list = [False, False]
<a name="cl-1567"></a>    cdef numpy.ndarray[numpy.npy_byte, ndim=2] cache_dirty = numpy.zeros(
<a name="cl-1568"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.byte)
<a name="cl-1569"></a>
<a name="cl-1570"></a>    block_col_size, block_row_size = dem_band.GetBlockSize()
<a name="cl-1571"></a>
<a name="cl-1572"></a>    cdef BlockCache_SWY block_cache = BlockCache_SWY(
<a name="cl-1573"></a>        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size,
<a name="cl-1574"></a>        block_col_size, band_list, block_list, update_list, cache_dirty)
<a name="cl-1575"></a>
<a name="cl-1576"></a>    cdef int n_global_block_rows = int(ceil(float(n_rows) / block_row_size))
<a name="cl-1577"></a>    cdef int n_global_block_cols = int(ceil(float(n_cols) / block_col_size))
<a name="cl-1578"></a>
<a name="cl-1579"></a>    cdef int global_row, global_col
<a name="cl-1580"></a>
<a name="cl-1581"></a>    cdef int cell_row_index, cell_col_index
<a name="cl-1582"></a>    cdef int cell_row_block_index, cell_col_block_index
<a name="cl-1583"></a>    cdef int cell_row_block_offset, cell_col_block_offset
<a name="cl-1584"></a>
<a name="cl-1585"></a>    cdef int neighbor_index
<a name="cl-1586"></a>    cdef int neighbor_row, neighbor_col
<a name="cl-1587"></a>    cdef int neighbor_row_index, neighbor_col_index
<a name="cl-1588"></a>    cdef int neighbor_row_block_offset, neighbor_col_block_offset
<a name="cl-1589"></a>
<a name="cl-1590"></a>    cdef float cell_dem, cell_flow, neighbor_dem, neighbor_flow
<a name="cl-1591"></a>
<a name="cl-1592"></a>    cdef float dem_nodata = pygeoprocessing.get_nodata_from_uri(
<a name="cl-1593"></a>        dem_uri)
<a name="cl-1594"></a>    cdef float flow_nodata = pygeoprocessing.get_nodata_from_uri(
<a name="cl-1595"></a>        flow_direction_uri)
<a name="cl-1596"></a>
<a name="cl-1597"></a>    cdef time_t last_time, current_time
<a name="cl-1598"></a>    time(&amp;last_time)
<a name="cl-1599"></a>
<a name="cl-1600"></a>    cdef neighbor_nodata
<a name="cl-1601"></a>
<a name="cl-1602"></a>    for global_block_row in xrange(n_global_block_rows):
<a name="cl-1603"></a>        time(&amp;current_time)
<a name="cl-1604"></a>        if current_time - last_time &gt; 5.0:
<a name="cl-1605"></a>            LOGGER.info(
<a name="cl-1606"></a>                "flat_edges %.1f%% complete", (global_row + 1.0) / n_rows * 100)
<a name="cl-1607"></a>            last_time = current_time
<a name="cl-1608"></a>        for global_block_col in xrange(n_global_block_cols):
<a name="cl-1609"></a>            for global_row in xrange(
<a name="cl-1610"></a>                    global_block_row*block_row_size,
<a name="cl-1611"></a>                    min((global_block_row+1)*block_row_size, n_rows)):
<a name="cl-1612"></a>                for global_col in xrange(
<a name="cl-1613"></a>                        global_block_col*block_col_size,
<a name="cl-1614"></a>                        min((global_block_col+1)*block_col_size, n_cols)):
<a name="cl-1615"></a>
<a name="cl-1616"></a>                    block_cache.update_cache(
<a name="cl-1617"></a>                        global_row, global_col,
<a name="cl-1618"></a>                        &amp;cell_row_index, &amp;cell_col_index,
<a name="cl-1619"></a>                        &amp;cell_row_block_offset, &amp;cell_col_block_offset)
<a name="cl-1620"></a>
<a name="cl-1621"></a>                    cell_dem = dem_block[cell_row_index, cell_col_index,
<a name="cl-1622"></a>                        cell_row_block_offset, cell_col_block_offset]
<a name="cl-1623"></a>
<a name="cl-1624"></a>                    if cell_dem == dem_nodata:
<a name="cl-1625"></a>                        continue
<a name="cl-1626"></a>
<a name="cl-1627"></a>                    cell_flow = flow_block[cell_row_index, cell_col_index,
<a name="cl-1628"></a>                        cell_row_block_offset, cell_col_block_offset]
<a name="cl-1629"></a>
<a name="cl-1630"></a>                    neighbor_nodata = 0
<a name="cl-1631"></a>                    for neighbor_index in xrange(8):
<a name="cl-1632"></a>                        neighbor_row = (
<a name="cl-1633"></a>                            neighbor_row_offset[neighbor_index] + global_row)
<a name="cl-1634"></a>                        neighbor_col = (
<a name="cl-1635"></a>                            neighbor_col_offset[neighbor_index] + global_col)
<a name="cl-1636"></a>
<a name="cl-1637"></a>                        if (neighbor_row &gt;= n_rows or neighbor_row &lt; 0 or
<a name="cl-1638"></a>                                neighbor_col &gt;= n_cols or neighbor_col &lt; 0):
<a name="cl-1639"></a>                            continue
<a name="cl-1640"></a>
<a name="cl-1641"></a>                        block_cache.update_cache(
<a name="cl-1642"></a>                            neighbor_row, neighbor_col,
<a name="cl-1643"></a>                            &amp;neighbor_row_index, &amp;neighbor_col_index,
<a name="cl-1644"></a>                            &amp;neighbor_row_block_offset,
<a name="cl-1645"></a>                            &amp;neighbor_col_block_offset)
<a name="cl-1646"></a>                        neighbor_dem = dem_block[
<a name="cl-1647"></a>                            neighbor_row_index, neighbor_col_index,
<a name="cl-1648"></a>                            neighbor_row_block_offset,
<a name="cl-1649"></a>                            neighbor_col_block_offset]
<a name="cl-1650"></a>
<a name="cl-1651"></a>                        if neighbor_dem == dem_nodata:
<a name="cl-1652"></a>                            neighbor_nodata = 1
<a name="cl-1653"></a>                            continue
<a name="cl-1654"></a>
<a name="cl-1655"></a>                        neighbor_flow = flow_block[
<a name="cl-1656"></a>                            neighbor_row_index, neighbor_col_index,
<a name="cl-1657"></a>                            neighbor_row_block_offset,
<a name="cl-1658"></a>                            neighbor_col_block_offset]
<a name="cl-1659"></a>
<a name="cl-1660"></a>                        if (cell_flow != flow_nodata and
<a name="cl-1661"></a>                                neighbor_flow == flow_nodata and
<a name="cl-1662"></a>                                cell_dem == neighbor_dem):
<a name="cl-1663"></a>                            low_edges.push_back(global_row * n_cols + global_col)
<a name="cl-1664"></a>                            break
<a name="cl-1665"></a>                        elif (cell_flow == flow_nodata and
<a name="cl-1666"></a>                              cell_dem &lt; neighbor_dem):
<a name="cl-1667"></a>                            high_edges.push_back(global_row * n_cols + global_col)
<a name="cl-1668"></a>                            break
<a name="cl-1669"></a>                    if drain_off_edge and neighbor_nodata:
<a name="cl-1670"></a>                        low_edges.push_back(global_row * n_cols + global_col)
<a name="cl-1671"></a>
<a name="cl-1672"></a>
<a name="cl-1673"></a>#@cython.boundscheck(False)
<a name="cl-1674"></a>@cython.wraparound(False)
<a name="cl-1675"></a>@cython.cdivision(True)
<a name="cl-1676"></a>cdef label_flats(dem_uri, deque[int] &amp;low_edges, labels_uri):
<a name="cl-1677"></a>    """A flood fill function to give all the cells of each flat a unique
<a name="cl-1678"></a>        label
<a name="cl-1679"></a>
<a name="cl-1680"></a>        Args:
<a name="cl-1681"></a>            dem_uri (string) - (input) a uri to a single band GDAL Dataset with
<a name="cl-1682"></a>                elevation values
<a name="cl-1683"></a>            low_edges (Set) - (input) Contains all the low edge cells of the dem
<a name="cl-1684"></a>                written as flat indexes in row major order
<a name="cl-1685"></a>            labels_uri (string) - (output) a uri to a single band integer gdal
<a name="cl-1686"></a>                dataset that will be created that will contain labels for the
<a name="cl-1687"></a>                flat regions of the DEM.
<a name="cl-1688"></a>            """
<a name="cl-1689"></a>
<a name="cl-1690"></a>    cdef int *neighbor_row_offset = [0, -1, -1, -1,  0,  1, 1, 1]
<a name="cl-1691"></a>    cdef int *neighbor_col_offset = [1,  1,  0, -1, -1, -1, 0, 1]
<a name="cl-1692"></a>
<a name="cl-1693"></a>    dem_ds = gdal.Open(dem_uri)
<a name="cl-1694"></a>    dem_band = dem_ds.GetRasterBand(1)
<a name="cl-1695"></a>
<a name="cl-1696"></a>    cdef int labels_nodata = -1
<a name="cl-1697"></a>    pygeoprocessing.new_raster_from_base_uri(
<a name="cl-1698"></a>        dem_uri, labels_uri, 'GTiff', labels_nodata,
<a name="cl-1699"></a>        gdal.GDT_Int32)
<a name="cl-1700"></a>    labels_ds = gdal.Open(labels_uri, gdal.GA_Update)
<a name="cl-1701"></a>    labels_band = labels_ds.GetRasterBand(1)
<a name="cl-1702"></a>
<a name="cl-1703"></a>    cdef int block_col_size, block_row_size
<a name="cl-1704"></a>    block_col_size, block_row_size = dem_band.GetBlockSize()
<a name="cl-1705"></a>    cdef int n_rows = dem_ds.RasterYSize
<a name="cl-1706"></a>    cdef int n_cols = dem_ds.RasterXSize
<a name="cl-1707"></a>
<a name="cl-1708"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] labels_block = numpy.zeros(
<a name="cl-1709"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
<a name="cl-1710"></a>        dtype=numpy.float32)
<a name="cl-1711"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] dem_block = numpy.zeros(
<a name="cl-1712"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
<a name="cl-1713"></a>        dtype=numpy.float32)
<a name="cl-1714"></a>
<a name="cl-1715"></a>    band_list = [dem_band, labels_band]
<a name="cl-1716"></a>    block_list = [dem_block, labels_block]
<a name="cl-1717"></a>    update_list = [False, True]
<a name="cl-1718"></a>    cdef numpy.ndarray[numpy.npy_byte, ndim=2] cache_dirty = numpy.zeros(
<a name="cl-1719"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.byte)
<a name="cl-1720"></a>
<a name="cl-1721"></a>    block_col_size, block_row_size = dem_band.GetBlockSize()
<a name="cl-1722"></a>
<a name="cl-1723"></a>    cdef BlockCache_SWY block_cache = BlockCache_SWY(
<a name="cl-1724"></a>        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size,
<a name="cl-1725"></a>        block_col_size, band_list, block_list, update_list, cache_dirty)
<a name="cl-1726"></a>
<a name="cl-1727"></a>    cdef int n_global_block_rows = int(ceil(float(n_rows) / block_row_size))
<a name="cl-1728"></a>    cdef int n_global_block_cols = int(ceil(float(n_cols) / block_col_size))
<a name="cl-1729"></a>
<a name="cl-1730"></a>    cdef int global_row, global_col
<a name="cl-1731"></a>
<a name="cl-1732"></a>    cdef int cell_row_index, cell_col_index
<a name="cl-1733"></a>    cdef int cell_row_block_index, cell_col_block_index
<a name="cl-1734"></a>    cdef int cell_row_block_offset, cell_col_block_offset
<a name="cl-1735"></a>
<a name="cl-1736"></a>    cdef int neighbor_index
<a name="cl-1737"></a>    cdef int neighbor_row, neighbor_col
<a name="cl-1738"></a>    cdef int neighbor_row_index, neighbor_col_index
<a name="cl-1739"></a>    cdef int neighbor_row_block_offset, neighbor_col_block_offset
<a name="cl-1740"></a>
<a name="cl-1741"></a>    cdef float cell_dem, neighbor_dem, neighbor_label
<a name="cl-1742"></a>    cdef float cell_label, flat_cell_label
<a name="cl-1743"></a>
<a name="cl-1744"></a>    cdef float dem_nodata = pygeoprocessing.get_nodata_from_uri(
<a name="cl-1745"></a>        dem_uri)
<a name="cl-1746"></a>
<a name="cl-1747"></a>    cdef time_t last_time, current_time
<a name="cl-1748"></a>    time(&amp;last_time)
<a name="cl-1749"></a>
<a name="cl-1750"></a>    cdef int flat_cell_index
<a name="cl-1751"></a>    cdef int flat_fill_cell_index
<a name="cl-1752"></a>    cdef int label = 1
<a name="cl-1753"></a>    cdef int fill_cell_row, fill_cell_col
<a name="cl-1754"></a>    cdef deque[int] to_fill
<a name="cl-1755"></a>    cdef float flat_height, current_flat_height
<a name="cl-1756"></a>    cdef int visit_number = 0
<a name="cl-1757"></a>    for _ in xrange(low_edges.size()):
<a name="cl-1758"></a>        flat_cell_index = low_edges.front()
<a name="cl-1759"></a>        low_edges.pop_front()
<a name="cl-1760"></a>        low_edges.push_back(flat_cell_index)
<a name="cl-1761"></a>        visit_number += 1
<a name="cl-1762"></a>        time(&amp;current_time)
<a name="cl-1763"></a>        if current_time - last_time &gt; 5.0:
<a name="cl-1764"></a>            LOGGER.info(
<a name="cl-1765"></a>                "label_flats %.1f%% complete",
<a name="cl-1766"></a>                float(visit_number) / low_edges.size() * 100)
<a name="cl-1767"></a>            last_time = current_time
<a name="cl-1768"></a>        global_row = flat_cell_index / n_cols
<a name="cl-1769"></a>        global_col = flat_cell_index % n_cols
<a name="cl-1770"></a>
<a name="cl-1771"></a>        block_cache.update_cache(
<a name="cl-1772"></a>            global_row, global_col,
<a name="cl-1773"></a>            &amp;cell_row_index, &amp;cell_col_index,
<a name="cl-1774"></a>            &amp;cell_row_block_offset, &amp;cell_col_block_offset)
<a name="cl-1775"></a>
<a name="cl-1776"></a>        cell_label = labels_block[cell_row_index, cell_col_index,
<a name="cl-1777"></a>            cell_row_block_offset, cell_col_block_offset]
<a name="cl-1778"></a>
<a name="cl-1779"></a>        flat_height = dem_block[cell_row_index, cell_col_index,
<a name="cl-1780"></a>            cell_row_block_offset, cell_col_block_offset]
<a name="cl-1781"></a>
<a name="cl-1782"></a>        if cell_label == labels_nodata:
<a name="cl-1783"></a>            #label flats
<a name="cl-1784"></a>            to_fill.push_back(flat_cell_index)
<a name="cl-1785"></a>            while not to_fill.empty():
<a name="cl-1786"></a>                flat_fill_cell_index = to_fill.front()
<a name="cl-1787"></a>                to_fill.pop_front()
<a name="cl-1788"></a>                fill_cell_row = flat_fill_cell_index / n_cols
<a name="cl-1789"></a>                fill_cell_col = flat_fill_cell_index % n_cols
<a name="cl-1790"></a>                if (fill_cell_row &lt; 0 or fill_cell_row &gt;= n_rows or
<a name="cl-1791"></a>                        fill_cell_col &lt; 0 or fill_cell_col &gt;= n_cols):
<a name="cl-1792"></a>                    continue
<a name="cl-1793"></a>
<a name="cl-1794"></a>                block_cache.update_cache(
<a name="cl-1795"></a>                    fill_cell_row, fill_cell_col,
<a name="cl-1796"></a>                    &amp;cell_row_index, &amp;cell_col_index,
<a name="cl-1797"></a>                    &amp;cell_row_block_offset, &amp;cell_col_block_offset)
<a name="cl-1798"></a>
<a name="cl-1799"></a>                current_flat_height = dem_block[cell_row_index, cell_col_index,
<a name="cl-1800"></a>                    cell_row_block_offset, cell_col_block_offset]
<a name="cl-1801"></a>
<a name="cl-1802"></a>                if current_flat_height != flat_height:
<a name="cl-1803"></a>                    continue
<a name="cl-1804"></a>
<a name="cl-1805"></a>                flat_cell_label = labels_block[
<a name="cl-1806"></a>                    cell_row_index, cell_col_index,
<a name="cl-1807"></a>                    cell_row_block_offset, cell_col_block_offset]
<a name="cl-1808"></a>
<a name="cl-1809"></a>                if flat_cell_label != labels_nodata:
<a name="cl-1810"></a>                    continue
<a name="cl-1811"></a>
<a name="cl-1812"></a>                #set the label
<a name="cl-1813"></a>                labels_block[
<a name="cl-1814"></a>                    cell_row_index, cell_col_index,
<a name="cl-1815"></a>                    cell_row_block_offset, cell_col_block_offset] = label
<a name="cl-1816"></a>                cache_dirty[cell_row_index, cell_col_index] = 1
<a name="cl-1817"></a>
<a name="cl-1818"></a>                #visit the neighbors
<a name="cl-1819"></a>                for neighbor_index in xrange(8):
<a name="cl-1820"></a>                    neighbor_row = (
<a name="cl-1821"></a>                        fill_cell_row + neighbor_row_offset[neighbor_index])
<a name="cl-1822"></a>                    neighbor_col = (
<a name="cl-1823"></a>                        fill_cell_col + neighbor_col_offset[neighbor_index])
<a name="cl-1824"></a>                    to_fill.push_back(neighbor_row * n_cols + neighbor_col)
<a name="cl-1825"></a>
<a name="cl-1826"></a>            label += 1
<a name="cl-1827"></a>    block_cache.flush_cache()
<a name="cl-1828"></a>
<a name="cl-1829"></a>
<a name="cl-1830"></a>#@cython.boundscheck(False)
<a name="cl-1831"></a>@cython.wraparound(False)
<a name="cl-1832"></a>@cython.cdivision(True)
<a name="cl-1833"></a>cdef clean_high_edges(labels_uri, deque[int] &amp;high_edges):
<a name="cl-1834"></a>    """Removes any high edges that do not have labels and reports them if so.
<a name="cl-1835"></a>
<a name="cl-1836"></a>        Args:
<a name="cl-1837"></a>            labels_uri (string) - (input) a uri to a single band integer gdal
<a name="cl-1838"></a>                dataset that contain labels for the cells that lie in
<a name="cl-1839"></a>                flat regions of the DEM.
<a name="cl-1840"></a>            high_edges (set) - (input/output) a set containing row major order
<a name="cl-1841"></a>                flat indexes
<a name="cl-1842"></a>
<a name="cl-1843"></a>        Returns:
<a name="cl-1844"></a>            nothing"""
<a name="cl-1845"></a>
<a name="cl-1846"></a>    labels_ds = gdal.Open(labels_uri)
<a name="cl-1847"></a>    labels_band = labels_ds.GetRasterBand(1)
<a name="cl-1848"></a>
<a name="cl-1849"></a>    cdef int block_col_size, block_row_size
<a name="cl-1850"></a>    block_col_size, block_row_size = labels_band.GetBlockSize()
<a name="cl-1851"></a>    cdef int n_rows = labels_ds.RasterYSize
<a name="cl-1852"></a>    cdef int n_cols = labels_ds.RasterXSize
<a name="cl-1853"></a>
<a name="cl-1854"></a>    cdef numpy.ndarray[numpy.npy_int32, ndim=4] labels_block = numpy.zeros(
<a name="cl-1855"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
<a name="cl-1856"></a>        dtype=numpy.int32)
<a name="cl-1857"></a>
<a name="cl-1858"></a>    band_list = [labels_band]
<a name="cl-1859"></a>    block_list = [labels_block]
<a name="cl-1860"></a>    update_list = [False]
<a name="cl-1861"></a>    cdef numpy.ndarray[numpy.npy_byte, ndim=2] cache_dirty = numpy.zeros(
<a name="cl-1862"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.byte)
<a name="cl-1863"></a>
<a name="cl-1864"></a>    cdef BlockCache_SWY block_cache = BlockCache_SWY(
<a name="cl-1865"></a>        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size,
<a name="cl-1866"></a>        block_col_size, band_list, block_list, update_list, cache_dirty)
<a name="cl-1867"></a>
<a name="cl-1868"></a>    cdef int labels_nodata = pygeoprocessing.get_nodata_from_uri(
<a name="cl-1869"></a>        labels_uri)
<a name="cl-1870"></a>    cdef int flat_cell_label
<a name="cl-1871"></a>
<a name="cl-1872"></a>    cdef int cell_row_index, cell_col_index
<a name="cl-1873"></a>    cdef int cell_row_block_index, cell_col_block_index
<a name="cl-1874"></a>    cdef int cell_row_block_offset, cell_col_block_offset
<a name="cl-1875"></a>
<a name="cl-1876"></a>    cdef int flat_index
<a name="cl-1877"></a>    cdef int flat_row, flat_col
<a name="cl-1878"></a>    cdef c_set[int] unlabeled_set
<a name="cl-1879"></a>    for _ in xrange(high_edges.size()):
<a name="cl-1880"></a>        flat_index = high_edges.front()
<a name="cl-1881"></a>        high_edges.pop_front()
<a name="cl-1882"></a>        high_edges.push_back(flat_index)
<a name="cl-1883"></a>        flat_row = flat_index / n_cols
<a name="cl-1884"></a>        flat_col = flat_index % n_cols
<a name="cl-1885"></a>
<a name="cl-1886"></a>        block_cache.update_cache(
<a name="cl-1887"></a>            flat_row, flat_col,
<a name="cl-1888"></a>            &amp;cell_row_index, &amp;cell_col_index,
<a name="cl-1889"></a>            &amp;cell_row_block_offset, &amp;cell_col_block_offset)
<a name="cl-1890"></a>
<a name="cl-1891"></a>        flat_cell_label = labels_block[
<a name="cl-1892"></a>            cell_row_index, cell_col_index,
<a name="cl-1893"></a>            cell_row_block_offset, cell_col_block_offset]
<a name="cl-1894"></a>
<a name="cl-1895"></a>        #this is a flat that does not have an outlet
<a name="cl-1896"></a>        if flat_cell_label == labels_nodata:
<a name="cl-1897"></a>            unlabeled_set.insert(flat_index)
<a name="cl-1898"></a>
<a name="cl-1899"></a>    if unlabeled_set.size() &gt; 0:
<a name="cl-1900"></a>        #remove high edges that are unlabeled
<a name="cl-1901"></a>        for _ in xrange(high_edges.size()):
<a name="cl-1902"></a>            flat_index = high_edges.front()
<a name="cl-1903"></a>            high_edges.pop_front()
<a name="cl-1904"></a>            if unlabeled_set.find(flat_index) != unlabeled_set.end():
<a name="cl-1905"></a>                high_edges.push_back(flat_index)
<a name="cl-1906"></a>        LOGGER.warn("Not all flats have outlets")
<a name="cl-1907"></a>    block_cache.flush_cache()
<a name="cl-1908"></a>
<a name="cl-1909"></a>
<a name="cl-1910"></a>#@cython.boundscheck(False)
<a name="cl-1911"></a>@cython.wraparound(False)
<a name="cl-1912"></a>@cython.cdivision(True)
<a name="cl-1913"></a>cdef drain_flats(
<a name="cl-1914"></a>        deque[int] &amp;high_edges, deque[int] &amp;low_edges, labels_uri,
<a name="cl-1915"></a>        flow_direction_uri, flat_mask_uri):
<a name="cl-1916"></a>    """A wrapper function for draining flats so it can be called from a
<a name="cl-1917"></a>        Python level, but use a C++ map at the Cython level.
<a name="cl-1918"></a>
<a name="cl-1919"></a>        Args:
<a name="cl-1920"></a>            high_edges (deque[int]) - (input) A list of row major order indicating the
<a name="cl-1921"></a>                high edge lists.
<a name="cl-1922"></a>            low_edges (deque[int]) - (input)  A list of row major order indicating the
<a name="cl-1923"></a>                high edge lists.
<a name="cl-1924"></a>            labels_uri (string) - (input) A uri to a gdal raster that has
<a name="cl-1925"></a>                unique integer labels for each flat in the DEM.
<a name="cl-1926"></a>            flow_direction_uri (string) - (input/output) A uri to a gdal raster
<a name="cl-1927"></a>                that has d-infinity flow directions defined for non-flat pixels
<a name="cl-1928"></a>                and will have pixels defined for the flat pixels when the
<a name="cl-1929"></a>                function returns
<a name="cl-1930"></a>            flat_mask_uri (string) - (out) A uri to a gdal raster that will have
<a name="cl-1931"></a>                relative heights defined per flat to drain each flat.
<a name="cl-1932"></a>
<a name="cl-1933"></a>        Returns:
<a name="cl-1934"></a>            nothing"""
<a name="cl-1935"></a>
<a name="cl-1936"></a>    cdef map[int, int] flat_height
<a name="cl-1937"></a>
<a name="cl-1938"></a>    LOGGER.info('draining away from higher')
<a name="cl-1939"></a>    away_from_higher(
<a name="cl-1940"></a>        high_edges, labels_uri, flow_direction_uri, flat_mask_uri, flat_height)
<a name="cl-1941"></a>
<a name="cl-1942"></a>    LOGGER.info('draining towards lower')
<a name="cl-1943"></a>    towards_lower(
<a name="cl-1944"></a>        low_edges, labels_uri, flow_direction_uri, flat_mask_uri, flat_height)
<a name="cl-1945"></a>
<a name="cl-1946"></a>
<a name="cl-1947"></a>#@cython.boundscheck(False)
<a name="cl-1948"></a>@cython.wraparound(False)
<a name="cl-1949"></a>@cython.cdivision(True)
<a name="cl-1950"></a>cdef away_from_higher(
<a name="cl-1951"></a>        deque[int] &amp;high_edges, labels_uri, flow_direction_uri, flat_mask_uri,
<a name="cl-1952"></a>        map[int, int] &amp;flat_height):
<a name="cl-1953"></a>    """Builds a gradient away from higher terrain.
<a name="cl-1954"></a>
<a name="cl-1955"></a>        Take Care, Take Care, Take Care
<a name="cl-1956"></a>        The Earth Is Not a Cold Dead Place
<a name="cl-1957"></a>        Those Who Tell The Truth Shall Die,
<a name="cl-1958"></a>            Those Who Tell The Truth Shall Live Forever
<a name="cl-1959"></a>
<a name="cl-1960"></a>        Args:
<a name="cl-1961"></a>            high_edges (deque) - (input) all the high edge cells of the DEM which
<a name="cl-1962"></a>                are part of drainable flats.
<a name="cl-1963"></a>            labels_uri (string) - (input) a uri to a single band integer gdal
<a name="cl-1964"></a>                dataset that contain labels for the cells that lie in
<a name="cl-1965"></a>                flat regions of the DEM.
<a name="cl-1966"></a>            flow_direction_uri (string) - (input) a uri to a single band
<a name="cl-1967"></a>                GDAL Dataset with partially defined d_infinity flow directions
<a name="cl-1968"></a>            flat_mask_uri (string) - (output) gdal dataset that contains the
<a name="cl-1969"></a>                number of increments to be applied to each cell to form a
<a name="cl-1970"></a>                gradient away from higher terrain.  cells not in a flat have a
<a name="cl-1971"></a>                value of 0
<a name="cl-1972"></a>            flat_height (collections.defaultdict) - (input/output) Has an entry
<a name="cl-1973"></a>                for each label value of of labels_uri indicating the maximal
<a name="cl-1974"></a>                number of increments to be applied to the flat idientifed by
<a name="cl-1975"></a>                that label.
<a name="cl-1976"></a>
<a name="cl-1977"></a>        Returns:
<a name="cl-1978"></a>            nothing"""
<a name="cl-1979"></a>
<a name="cl-1980"></a>    cdef int *neighbor_row_offset = [0, -1, -1, -1,  0,  1, 1, 1]
<a name="cl-1981"></a>    cdef int *neighbor_col_offset = [1,  1,  0, -1, -1, -1, 0, 1]
<a name="cl-1982"></a>
<a name="cl-1983"></a>    cdef int flat_mask_nodata = -9999
<a name="cl-1984"></a>    #fill up the flat mask with 0s so it can be used to route a dem later
<a name="cl-1985"></a>    pygeoprocessing.new_raster_from_base_uri(
<a name="cl-1986"></a>        labels_uri, flat_mask_uri, 'GTiff', flat_mask_nodata,
<a name="cl-1987"></a>        gdal.GDT_Int32, fill_value=0)
<a name="cl-1988"></a>
<a name="cl-1989"></a>    labels_ds = gdal.Open(labels_uri)
<a name="cl-1990"></a>    labels_band = labels_ds.GetRasterBand(1)
<a name="cl-1991"></a>    flat_mask_ds = gdal.Open(flat_mask_uri, gdal.GA_Update)
<a name="cl-1992"></a>    flat_mask_band = flat_mask_ds.GetRasterBand(1)
<a name="cl-1993"></a>    flow_direction_ds = gdal.Open(flow_direction_uri)
<a name="cl-1994"></a>    flow_direction_band = flow_direction_ds.GetRasterBand(1)
<a name="cl-1995"></a>
<a name="cl-1996"></a>    cdef int block_col_size, block_row_size
<a name="cl-1997"></a>    block_col_size, block_row_size = labels_band.GetBlockSize()
<a name="cl-1998"></a>    cdef int n_rows = labels_ds.RasterYSize
<a name="cl-1999"></a>    cdef int n_cols = labels_ds.RasterXSize
<a name="cl-2000"></a>
<a name="cl-2001"></a>    cdef numpy.ndarray[numpy.npy_int32, ndim=4] labels_block = numpy.zeros(
<a name="cl-2002"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
<a name="cl-2003"></a>        dtype=numpy.int32)
<a name="cl-2004"></a>    cdef numpy.ndarray[numpy.npy_int32, ndim=4] flat_mask_block = numpy.zeros(
<a name="cl-2005"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
<a name="cl-2006"></a>        dtype=numpy.int32)
<a name="cl-2007"></a>    cdef numpy.ndarray[numpy.npy_int32, ndim=4] flow_direction_block = (
<a name="cl-2008"></a>        numpy.zeros(
<a name="cl-2009"></a>            (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
<a name="cl-2010"></a>            dtype=numpy.int32))
<a name="cl-2011"></a>
<a name="cl-2012"></a>    band_list = [labels_band, flat_mask_band, flow_direction_band]
<a name="cl-2013"></a>    block_list = [labels_block, flat_mask_block, flow_direction_block]
<a name="cl-2014"></a>    update_list = [False, True, False]
<a name="cl-2015"></a>    cdef numpy.ndarray[numpy.npy_byte, ndim=2] cache_dirty = numpy.zeros(
<a name="cl-2016"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.byte)
<a name="cl-2017"></a>
<a name="cl-2018"></a>    cdef BlockCache_SWY block_cache = BlockCache_SWY(
<a name="cl-2019"></a>        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size,
<a name="cl-2020"></a>        block_col_size, band_list, block_list, update_list, cache_dirty)
<a name="cl-2021"></a>
<a name="cl-2022"></a>    cdef int cell_row_index, cell_col_index
<a name="cl-2023"></a>    cdef int cell_row_block_index, cell_col_block_index
<a name="cl-2024"></a>    cdef int cell_row_block_offset, cell_col_block_offset
<a name="cl-2025"></a>
<a name="cl-2026"></a>    cdef int loops = 1
<a name="cl-2027"></a>
<a name="cl-2028"></a>    cdef int neighbor_row, neighbor_col
<a name="cl-2029"></a>    cdef int flat_index
<a name="cl-2030"></a>    cdef int flat_row, flat_col
<a name="cl-2031"></a>    cdef int flat_mask
<a name="cl-2032"></a>    cdef int labels_nodata = pygeoprocessing.get_nodata_from_uri(labels_uri)
<a name="cl-2033"></a>    cdef int cell_label, neighbor_label
<a name="cl-2034"></a>    cdef float neighbor_flow
<a name="cl-2035"></a>    cdef float flow_nodata = pygeoprocessing.get_nodata_from_uri(
<a name="cl-2036"></a>        flow_direction_uri)
<a name="cl-2037"></a>
<a name="cl-2038"></a>    cdef time_t last_time, current_time
<a name="cl-2039"></a>    time(&amp;last_time)
<a name="cl-2040"></a>
<a name="cl-2041"></a>    cdef deque[int] high_edges_queue
<a name="cl-2042"></a>
<a name="cl-2043"></a>    #seed the queue with the high edges
<a name="cl-2044"></a>    for _ in xrange(high_edges.size()):
<a name="cl-2045"></a>        flat_index = high_edges.front()
<a name="cl-2046"></a>        high_edges.pop_front()
<a name="cl-2047"></a>        high_edges.push_back(flat_index)
<a name="cl-2048"></a>        high_edges_queue.push_back(flat_index)
<a name="cl-2049"></a>
<a name="cl-2050"></a>    marker = -1
<a name="cl-2051"></a>    high_edges_queue.push_back(marker)
<a name="cl-2052"></a>
<a name="cl-2053"></a>    while high_edges_queue.size() &gt; 1:
<a name="cl-2054"></a>        time(&amp;current_time)
<a name="cl-2055"></a>        if current_time - last_time &gt; 5.0:
<a name="cl-2056"></a>            LOGGER.info(
<a name="cl-2057"></a>                "away_from_higher, work queue size: %d complete",
<a name="cl-2058"></a>                high_edges_queue.size())
<a name="cl-2059"></a>            last_time = current_time
<a name="cl-2060"></a>
<a name="cl-2061"></a>        flat_index = high_edges_queue.front()
<a name="cl-2062"></a>        high_edges_queue.pop_front()
<a name="cl-2063"></a>        if flat_index == marker:
<a name="cl-2064"></a>            loops += 1
<a name="cl-2065"></a>            high_edges_queue.push_back(marker)
<a name="cl-2066"></a>            continue
<a name="cl-2067"></a>
<a name="cl-2068"></a>        flat_row = flat_index / n_cols
<a name="cl-2069"></a>        flat_col = flat_index % n_cols
<a name="cl-2070"></a>
<a name="cl-2071"></a>        block_cache.update_cache(
<a name="cl-2072"></a>            flat_row, flat_col,
<a name="cl-2073"></a>            &amp;cell_row_index, &amp;cell_col_index,
<a name="cl-2074"></a>            &amp;cell_row_block_offset, &amp;cell_col_block_offset)
<a name="cl-2075"></a>
<a name="cl-2076"></a>        flat_mask = flat_mask_block[
<a name="cl-2077"></a>            cell_row_index, cell_col_index,
<a name="cl-2078"></a>            cell_row_block_offset, cell_col_block_offset]
<a name="cl-2079"></a>
<a name="cl-2080"></a>        cell_label = labels_block[
<a name="cl-2081"></a>            cell_row_index, cell_col_index,
<a name="cl-2082"></a>            cell_row_block_offset, cell_col_block_offset]
<a name="cl-2083"></a>
<a name="cl-2084"></a>        if flat_mask != 0:
<a name="cl-2085"></a>            continue
<a name="cl-2086"></a>
<a name="cl-2087"></a>        #update the cell mask and the max height of the flat
<a name="cl-2088"></a>        #making it negative because it's easier to do here than in towards lower
<a name="cl-2089"></a>        flat_mask_block[
<a name="cl-2090"></a>            cell_row_index, cell_col_index,
<a name="cl-2091"></a>            cell_row_block_offset, cell_col_block_offset] = -loops
<a name="cl-2092"></a>        cache_dirty[cell_row_index, cell_col_index] = 1
<a name="cl-2093"></a>        flat_height[cell_label] = loops
<a name="cl-2094"></a>
<a name="cl-2095"></a>        #visit the neighbors
<a name="cl-2096"></a>        for neighbor_index in xrange(8):
<a name="cl-2097"></a>            neighbor_row = (
<a name="cl-2098"></a>                flat_row + neighbor_row_offset[neighbor_index])
<a name="cl-2099"></a>            neighbor_col = (
<a name="cl-2100"></a>                flat_col + neighbor_col_offset[neighbor_index])
<a name="cl-2101"></a>
<a name="cl-2102"></a>            if (neighbor_row &lt; 0 or neighbor_row &gt;= n_rows or
<a name="cl-2103"></a>                    neighbor_col &lt; 0 or neighbor_col &gt;= n_cols):
<a name="cl-2104"></a>                continue
<a name="cl-2105"></a>
<a name="cl-2106"></a>            block_cache.update_cache(
<a name="cl-2107"></a>                neighbor_row, neighbor_col,
<a name="cl-2108"></a>                &amp;cell_row_index, &amp;cell_col_index,
<a name="cl-2109"></a>                &amp;cell_row_block_offset, &amp;cell_col_block_offset)
<a name="cl-2110"></a>
<a name="cl-2111"></a>            neighbor_label = labels_block[
<a name="cl-2112"></a>                cell_row_index, cell_col_index,
<a name="cl-2113"></a>                cell_row_block_offset, cell_col_block_offset]
<a name="cl-2114"></a>
<a name="cl-2115"></a>            neighbor_flow = flow_direction_block[
<a name="cl-2116"></a>                cell_row_index, cell_col_index,
<a name="cl-2117"></a>                cell_row_block_offset, cell_col_block_offset]
<a name="cl-2118"></a>
<a name="cl-2119"></a>            if (neighbor_label != labels_nodata and
<a name="cl-2120"></a>                    neighbor_label == cell_label and
<a name="cl-2121"></a>                    neighbor_flow == flow_nodata):
<a name="cl-2122"></a>                high_edges_queue.push_back(neighbor_row * n_cols + neighbor_col)
<a name="cl-2123"></a>
<a name="cl-2124"></a>    block_cache.flush_cache()
<a name="cl-2125"></a>
<a name="cl-2126"></a>
<a name="cl-2127"></a>#@cython.boundscheck(False)
<a name="cl-2128"></a>@cython.wraparound(False)
<a name="cl-2129"></a>@cython.cdivision(True)
<a name="cl-2130"></a>cdef towards_lower(
<a name="cl-2131"></a>        deque[int] &amp;low_edges, labels_uri, flow_direction_uri, flat_mask_uri,
<a name="cl-2132"></a>        map[int, int] &amp;flat_height):
<a name="cl-2133"></a>    """Builds a gradient towards lower terrain.
<a name="cl-2134"></a>
<a name="cl-2135"></a>        Args:
<a name="cl-2136"></a>            low_edges (set) - (input) all the low edge cells of the DEM which
<a name="cl-2137"></a>                are part of drainable flats.
<a name="cl-2138"></a>            labels_uri (string) - (input) a uri to a single band integer gdal
<a name="cl-2139"></a>                dataset that contain labels for the cells that lie in
<a name="cl-2140"></a>                flat regions of the DEM.
<a name="cl-2141"></a>            flow_direction_uri (string) - (input) a uri to a single band
<a name="cl-2142"></a>                GDAL Dataset with partially defined d_infinity flow directions
<a name="cl-2143"></a>            flat_mask_uri (string) - (input/output) gdal dataset that contains
<a name="cl-2144"></a>                the negative step increments from toward_higher and will contain
<a name="cl-2145"></a>                the number of steps to be applied to each cell to form a
<a name="cl-2146"></a>                gradient away from higher terrain.  cells not in a flat have a
<a name="cl-2147"></a>                value of 0
<a name="cl-2148"></a>            flat_height (collections.defaultdict) - (input/output) Has an entry
<a name="cl-2149"></a>                for each label value of of labels_uri indicating the maximal
<a name="cl-2150"></a>                number of increments to be applied to the flat idientifed by
<a name="cl-2151"></a>                that label.
<a name="cl-2152"></a>
<a name="cl-2153"></a>        Returns:
<a name="cl-2154"></a>            nothing"""
<a name="cl-2155"></a>
<a name="cl-2156"></a>    cdef int *neighbor_row_offset = [0, -1, -1, -1,  0,  1, 1, 1]
<a name="cl-2157"></a>    cdef int *neighbor_col_offset = [1,  1,  0, -1, -1, -1, 0, 1]
<a name="cl-2158"></a>
<a name="cl-2159"></a>    flat_mask_nodata = pygeoprocessing.get_nodata_from_uri(flat_mask_uri)
<a name="cl-2160"></a>
<a name="cl-2161"></a>    labels_ds = gdal.Open(labels_uri)
<a name="cl-2162"></a>    labels_band = labels_ds.GetRasterBand(1)
<a name="cl-2163"></a>    flat_mask_ds = gdal.Open(flat_mask_uri, gdal.GA_Update)
<a name="cl-2164"></a>    flat_mask_band = flat_mask_ds.GetRasterBand(1)
<a name="cl-2165"></a>    flow_direction_ds = gdal.Open(flow_direction_uri)
<a name="cl-2166"></a>    flow_direction_band = flow_direction_ds.GetRasterBand(1)
<a name="cl-2167"></a>
<a name="cl-2168"></a>    cdef int block_col_size, block_row_size
<a name="cl-2169"></a>    block_col_size, block_row_size = labels_band.GetBlockSize()
<a name="cl-2170"></a>    cdef int n_rows = labels_ds.RasterYSize
<a name="cl-2171"></a>    cdef int n_cols = labels_ds.RasterXSize
<a name="cl-2172"></a>
<a name="cl-2173"></a>    cdef numpy.ndarray[numpy.npy_int32, ndim=4] labels_block = numpy.zeros(
<a name="cl-2174"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
<a name="cl-2175"></a>        dtype=numpy.int32)
<a name="cl-2176"></a>    cdef numpy.ndarray[numpy.npy_int32, ndim=4] flat_mask_block = numpy.zeros(
<a name="cl-2177"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
<a name="cl-2178"></a>        dtype=numpy.int32)
<a name="cl-2179"></a>    cdef numpy.ndarray[numpy.npy_int32, ndim=4] flow_direction_block = (
<a name="cl-2180"></a>        numpy.zeros(
<a name="cl-2181"></a>            (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
<a name="cl-2182"></a>            dtype=numpy.int32))
<a name="cl-2183"></a>
<a name="cl-2184"></a>    band_list = [labels_band, flat_mask_band, flow_direction_band]
<a name="cl-2185"></a>    block_list = [labels_block, flat_mask_block, flow_direction_block]
<a name="cl-2186"></a>    update_list = [False, True, False]
<a name="cl-2187"></a>    cdef numpy.ndarray[numpy.npy_byte, ndim=2] cache_dirty = numpy.zeros(
<a name="cl-2188"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.byte)
<a name="cl-2189"></a>
<a name="cl-2190"></a>    cdef BlockCache_SWY block_cache = BlockCache_SWY(
<a name="cl-2191"></a>        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size,
<a name="cl-2192"></a>        block_col_size, band_list, block_list, update_list, cache_dirty)
<a name="cl-2193"></a>
<a name="cl-2194"></a>    cdef int cell_row_index, cell_col_index
<a name="cl-2195"></a>    cdef int cell_row_block_index, cell_col_block_index
<a name="cl-2196"></a>    cdef int cell_row_block_offset, cell_col_block_offset
<a name="cl-2197"></a>
<a name="cl-2198"></a>    cdef int loops = 1
<a name="cl-2199"></a>
<a name="cl-2200"></a>    cdef deque[int] low_edges_queue
<a name="cl-2201"></a>    cdef int neighbor_row, neighbor_col
<a name="cl-2202"></a>    cdef int flat_index
<a name="cl-2203"></a>    cdef int flat_row, flat_col
<a name="cl-2204"></a>    cdef int flat_mask
<a name="cl-2205"></a>    cdef int labels_nodata = pygeoprocessing.get_nodata_from_uri(labels_uri)
<a name="cl-2206"></a>    cdef int cell_label, neighbor_label
<a name="cl-2207"></a>    cdef float neighbor_flow
<a name="cl-2208"></a>    cdef float flow_nodata = pygeoprocessing.get_nodata_from_uri(
<a name="cl-2209"></a>        flow_direction_uri)
<a name="cl-2210"></a>
<a name="cl-2211"></a>    #seed the queue with the low edges
<a name="cl-2212"></a>    for _ in xrange(low_edges.size()):
<a name="cl-2213"></a>        flat_index = low_edges.front()
<a name="cl-2214"></a>        low_edges.pop_front()
<a name="cl-2215"></a>        low_edges.push_back(flat_index)
<a name="cl-2216"></a>        low_edges_queue.push_back(flat_index)
<a name="cl-2217"></a>
<a name="cl-2218"></a>    cdef time_t last_time, current_time
<a name="cl-2219"></a>    time(&amp;last_time)
<a name="cl-2220"></a>
<a name="cl-2221"></a>    marker = -1
<a name="cl-2222"></a>    low_edges_queue.push_back(marker)
<a name="cl-2223"></a>    while low_edges_queue.size() &gt; 1:
<a name="cl-2224"></a>
<a name="cl-2225"></a>        time(&amp;current_time)
<a name="cl-2226"></a>        if current_time - last_time &gt; 5.0:
<a name="cl-2227"></a>            LOGGER.info(
<a name="cl-2228"></a>                "toward_lower work queue size: %d", low_edges_queue.size())
<a name="cl-2229"></a>            last_time = current_time
<a name="cl-2230"></a>
<a name="cl-2231"></a>        flat_index = low_edges_queue.front()
<a name="cl-2232"></a>        low_edges_queue.pop_front()
<a name="cl-2233"></a>        if flat_index == marker:
<a name="cl-2234"></a>            loops += 1
<a name="cl-2235"></a>            low_edges_queue.push_back(marker)
<a name="cl-2236"></a>            continue
<a name="cl-2237"></a>
<a name="cl-2238"></a>        flat_row = flat_index / n_cols
<a name="cl-2239"></a>        flat_col = flat_index % n_cols
<a name="cl-2240"></a>
<a name="cl-2241"></a>        block_cache.update_cache(
<a name="cl-2242"></a>            flat_row, flat_col,
<a name="cl-2243"></a>            &amp;cell_row_index, &amp;cell_col_index,
<a name="cl-2244"></a>            &amp;cell_row_block_offset, &amp;cell_col_block_offset)
<a name="cl-2245"></a>
<a name="cl-2246"></a>        flat_mask = flat_mask_block[
<a name="cl-2247"></a>            cell_row_index, cell_col_index,
<a name="cl-2248"></a>            cell_row_block_offset, cell_col_block_offset]
<a name="cl-2249"></a>
<a name="cl-2250"></a>        if flat_mask &gt; 0:
<a name="cl-2251"></a>            continue
<a name="cl-2252"></a>
<a name="cl-2253"></a>        cell_label = labels_block[
<a name="cl-2254"></a>            cell_row_index, cell_col_index,
<a name="cl-2255"></a>            cell_row_block_offset, cell_col_block_offset]
<a name="cl-2256"></a>
<a name="cl-2257"></a>        if flat_mask &lt; 0:
<a name="cl-2258"></a>            flat_mask_block[
<a name="cl-2259"></a>                cell_row_index, cell_col_index,
<a name="cl-2260"></a>                cell_row_block_offset, cell_col_block_offset] = (
<a name="cl-2261"></a>                    flat_height[cell_label] + flat_mask + 2 * loops)
<a name="cl-2262"></a>        else:
<a name="cl-2263"></a>            flat_mask_block[
<a name="cl-2264"></a>                cell_row_index, cell_col_index,
<a name="cl-2265"></a>                cell_row_block_offset, cell_col_block_offset] = 2 * loops
<a name="cl-2266"></a>        cache_dirty[cell_row_index, cell_col_index] = 1
<a name="cl-2267"></a>
<a name="cl-2268"></a>        #visit the neighbors
<a name="cl-2269"></a>        for neighbor_index in xrange(8):
<a name="cl-2270"></a>            neighbor_row = (
<a name="cl-2271"></a>                flat_row + neighbor_row_offset[neighbor_index])
<a name="cl-2272"></a>            neighbor_col = (
<a name="cl-2273"></a>                flat_col + neighbor_col_offset[neighbor_index])
<a name="cl-2274"></a>
<a name="cl-2275"></a>            if (neighbor_row &lt; 0 or neighbor_row &gt;= n_rows or
<a name="cl-2276"></a>                    neighbor_col &lt; 0 or neighbor_col &gt;= n_cols):
<a name="cl-2277"></a>                continue
<a name="cl-2278"></a>
<a name="cl-2279"></a>            block_cache.update_cache(
<a name="cl-2280"></a>                neighbor_row, neighbor_col,
<a name="cl-2281"></a>                &amp;cell_row_index, &amp;cell_col_index,
<a name="cl-2282"></a>                &amp;cell_row_block_offset, &amp;cell_col_block_offset)
<a name="cl-2283"></a>
<a name="cl-2284"></a>            neighbor_label = labels_block[
<a name="cl-2285"></a>                cell_row_index, cell_col_index,
<a name="cl-2286"></a>                cell_row_block_offset, cell_col_block_offset]
<a name="cl-2287"></a>
<a name="cl-2288"></a>            neighbor_flow = flow_direction_block[
<a name="cl-2289"></a>                cell_row_index, cell_col_index,
<a name="cl-2290"></a>                cell_row_block_offset, cell_col_block_offset]
<a name="cl-2291"></a>
<a name="cl-2292"></a>            if (neighbor_label != labels_nodata and
<a name="cl-2293"></a>                    neighbor_label == cell_label and
<a name="cl-2294"></a>                    neighbor_flow == flow_nodata):
<a name="cl-2295"></a>                low_edges_queue.push_back(neighbor_row * n_cols + neighbor_col)
<a name="cl-2296"></a>
<a name="cl-2297"></a>    block_cache.flush_cache()
<a name="cl-2298"></a>
<a name="cl-2299"></a>
<a name="cl-2300"></a>#@cython.boundscheck(False)
<a name="cl-2301"></a>@cython.wraparound(False)
<a name="cl-2302"></a>@cython.cdivision(True)
<a name="cl-2303"></a>def flow_direction_inf_masked_flow_dirs(
<a name="cl-2304"></a>        flat_mask_uri, labels_uri, flow_direction_uri):
<a name="cl-2305"></a>    """Calculates the D-infinity flow algorithm for regions defined from flat
<a name="cl-2306"></a>        drainage resolution.
<a name="cl-2307"></a>
<a name="cl-2308"></a>        Flow algorithm from: Tarboton, "A new method for the determination of
<a name="cl-2309"></a>        flow directions and upslope areas in grid digital elevation models,"
<a name="cl-2310"></a>        Water Resources Research, vol. 33, no. 2, pages 309 - 319, February
<a name="cl-2311"></a>        1997.
<a name="cl-2312"></a>
<a name="cl-2313"></a>        Also resolves flow directions in flat areas of DEM.
<a name="cl-2314"></a>
<a name="cl-2315"></a>        flat_mask_uri (string) - (input) a uri to a single band GDAL Dataset
<a name="cl-2316"></a>            that has offset values from the flat region resolution algorithm.
<a name="cl-2317"></a>            The offsets in flat_mask are the relative heights only within the
<a name="cl-2318"></a>            flat regions defined in labels_uri.
<a name="cl-2319"></a>        labels_uri (string) - (input) a uri to a single band integer gdal
<a name="cl-2320"></a>                dataset that contain labels for the cells that lie in
<a name="cl-2321"></a>                flat regions of the DEM.
<a name="cl-2322"></a>        flow_direction_uri - (input/output) a uri to an existing GDAL dataset
<a name="cl-2323"></a>            of same size as dem_uri.  Flow direction will be defined in regions
<a name="cl-2324"></a>            that have nodata values in them that overlap regions of labels_uri.
<a name="cl-2325"></a>            This is so this function can be used as a two pass filter for
<a name="cl-2326"></a>            resolving flow directions on a raw dem, then filling plateaus and
<a name="cl-2327"></a>            doing another pass.
<a name="cl-2328"></a>
<a name="cl-2329"></a>       returns nothing"""
<a name="cl-2330"></a>
<a name="cl-2331"></a>    cdef int col_index, row_index, n_cols, n_rows, max_index, facet_index, flat_index
<a name="cl-2332"></a>    cdef double e_0, e_1, e_2, s_1, s_2, d_1, d_2, flow_direction, slope, \
<a name="cl-2333"></a>        flow_direction_max_slope, slope_max, nodata_flow
<a name="cl-2334"></a>
<a name="cl-2335"></a>    flat_mask_ds = gdal.Open(flat_mask_uri)
<a name="cl-2336"></a>    flat_mask_band = flat_mask_ds.GetRasterBand(1)
<a name="cl-2337"></a>
<a name="cl-2338"></a>    #facet elevation and factors for slope and flow_direction calculations
<a name="cl-2339"></a>    #from Table 1 in Tarboton 1997.
<a name="cl-2340"></a>    #THIS IS IMPORTANT:  The order is row (j), column (i), transposed to GDAL
<a name="cl-2341"></a>    #convention.
<a name="cl-2342"></a>    cdef int *e_0_offsets = [+0, +0,
<a name="cl-2343"></a>                             +0, +0,
<a name="cl-2344"></a>                             +0, +0,
<a name="cl-2345"></a>                             +0, +0,
<a name="cl-2346"></a>                             +0, +0,
<a name="cl-2347"></a>                             +0, +0,
<a name="cl-2348"></a>                             +0, +0,
<a name="cl-2349"></a>                             +0, +0]
<a name="cl-2350"></a>    cdef int *e_1_offsets = [+0, +1,
<a name="cl-2351"></a>                             -1, +0,
<a name="cl-2352"></a>                             -1, +0,
<a name="cl-2353"></a>                             +0, -1,
<a name="cl-2354"></a>                             +0, -1,
<a name="cl-2355"></a>                             +1, +0,
<a name="cl-2356"></a>                             +1, +0,
<a name="cl-2357"></a>                             +0, +1]
<a name="cl-2358"></a>    cdef int *e_2_offsets = [-1, +1,
<a name="cl-2359"></a>                             -1, +1,
<a name="cl-2360"></a>                             -1, -1,
<a name="cl-2361"></a>                             -1, -1,
<a name="cl-2362"></a>                             +1, -1,
<a name="cl-2363"></a>                             +1, -1,
<a name="cl-2364"></a>                             +1, +1,
<a name="cl-2365"></a>                             +1, +1]
<a name="cl-2366"></a>    cdef int *a_c = [0, 1, 1, 2, 2, 3, 3, 4]
<a name="cl-2367"></a>    cdef int *a_f = [1, -1, 1, -1, 1, -1, 1, -1]
<a name="cl-2368"></a>
<a name="cl-2369"></a>    cdef int *row_offsets = [0, -1, -1, -1,  0,  1, 1, 1]
<a name="cl-2370"></a>    cdef int *col_offsets = [1,  1,  0, -1, -1, -1, 0, 1]
<a name="cl-2371"></a>
<a name="cl-2372"></a>    n_rows, n_cols = pygeoprocessing.get_row_col_from_uri(flat_mask_uri)
<a name="cl-2373"></a>    d_1 = pygeoprocessing.get_cell_size_from_uri(flat_mask_uri)
<a name="cl-2374"></a>    d_2 = d_1
<a name="cl-2375"></a>    cdef double max_r = numpy.pi / 4.0
<a name="cl-2376"></a>
<a name="cl-2377"></a>
<a name="cl-2378"></a>    cdef float flow_nodata = pygeoprocessing.get_nodata_from_uri(
<a name="cl-2379"></a>        flow_direction_uri)
<a name="cl-2380"></a>    flow_direction_dataset = gdal.Open(flow_direction_uri, gdal.GA_Update)
<a name="cl-2381"></a>    flow_band = flow_direction_dataset.GetRasterBand(1)
<a name="cl-2382"></a>
<a name="cl-2383"></a>    cdef float label_nodata = pygeoprocessing.get_nodata_from_uri(labels_uri)
<a name="cl-2384"></a>    label_dataset = gdal.Open(labels_uri)
<a name="cl-2385"></a>    label_band = label_dataset.GetRasterBand(1)
<a name="cl-2386"></a>
<a name="cl-2387"></a>    #center point of global index
<a name="cl-2388"></a>    cdef int block_row_size, block_col_size
<a name="cl-2389"></a>    block_col_size, block_row_size = flat_mask_band.GetBlockSize()
<a name="cl-2390"></a>    cdef int global_row, global_col, e_0_row, e_0_col, e_1_row, e_1_col, e_2_row, e_2_col #index into the overall raster
<a name="cl-2391"></a>    cdef int e_0_row_index, e_0_col_index #the index of the cache block
<a name="cl-2392"></a>    cdef int e_0_row_block_offset, e_0_col_block_offset #index into the cache block
<a name="cl-2393"></a>    cdef int e_1_row_index, e_1_col_index #the index of the cache block
<a name="cl-2394"></a>    cdef int e_1_row_block_offset, e_1_col_block_offset #index into the cache block
<a name="cl-2395"></a>    cdef int e_2_row_index, e_2_col_index #the index of the cache block
<a name="cl-2396"></a>    cdef int e_2_row_block_offset, e_2_col_block_offset #index into the cache block
<a name="cl-2397"></a>
<a name="cl-2398"></a>    cdef int global_block_row, global_block_col #used to walk the global blocks
<a name="cl-2399"></a>
<a name="cl-2400"></a>    #neighbor sections of global index
<a name="cl-2401"></a>    cdef int neighbor_row, neighbor_col #neighbor equivalent of global_{row,col}
<a name="cl-2402"></a>    cdef int neighbor_row_index, neighbor_col_index #neighbor cache index
<a name="cl-2403"></a>    cdef int neighbor_row_block_offset, neighbor_col_block_offset #index into the neighbor cache block
<a name="cl-2404"></a>
<a name="cl-2405"></a>    #define all the caches
<a name="cl-2406"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] flow_block = numpy.zeros(
<a name="cl-2407"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-2408"></a>    #flat_mask block is a 64 bit float so it can capture the resolution of small flat_mask offsets
<a name="cl-2409"></a>    #from the plateau resolution algorithm.
<a name="cl-2410"></a>    cdef numpy.ndarray[numpy.npy_int32, ndim=4] flat_mask_block = numpy.zeros(
<a name="cl-2411"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.int32)
<a name="cl-2412"></a>    cdef numpy.ndarray[numpy.npy_int32, ndim=4] label_block = numpy.zeros(
<a name="cl-2413"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.int32)
<a name="cl-2414"></a>
<a name="cl-2415"></a>    #the BlockCache_SWY object needs parallel lists of bands, blocks, and boolean tags to indicate which ones are updated
<a name="cl-2416"></a>    band_list = [flat_mask_band, flow_band, label_band]
<a name="cl-2417"></a>    block_list = [flat_mask_block, flow_block, label_block]
<a name="cl-2418"></a>    update_list = [False, True, False]
<a name="cl-2419"></a>    cdef numpy.ndarray[numpy.npy_byte, ndim=2] cache_dirty = numpy.zeros((N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.byte)
<a name="cl-2420"></a>
<a name="cl-2421"></a>    cdef BlockCache_SWY block_cache = BlockCache_SWY(
<a name="cl-2422"></a>        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size, block_col_size, band_list, block_list, update_list, cache_dirty)
<a name="cl-2423"></a>
<a name="cl-2424"></a>    cdef int row_offset, col_offset
<a name="cl-2425"></a>
<a name="cl-2426"></a>    cdef int n_global_block_rows = int(ceil(float(n_rows) / block_row_size))
<a name="cl-2427"></a>    cdef int n_global_block_cols = int(ceil(float(n_cols) / block_col_size))
<a name="cl-2428"></a>    cdef time_t last_time, current_time
<a name="cl-2429"></a>    cdef float current_flow
<a name="cl-2430"></a>    cdef int current_label, e_1_label, e_2_label
<a name="cl-2431"></a>    time(&amp;last_time)
<a name="cl-2432"></a>    #flow not defined on the edges, so just go 1 row in
<a name="cl-2433"></a>    for global_block_row in xrange(n_global_block_rows):
<a name="cl-2434"></a>        time(&amp;current_time)
<a name="cl-2435"></a>        if current_time - last_time &gt; 5.0:
<a name="cl-2436"></a>            LOGGER.info("flow_direction_inf %.1f%% complete", (global_row + 1.0) / n_rows * 100)
<a name="cl-2437"></a>            last_time = current_time
<a name="cl-2438"></a>        for global_block_col in xrange(n_global_block_cols):
<a name="cl-2439"></a>            for global_row in xrange(global_block_row*block_row_size, min((global_block_row+1)*block_row_size, n_rows)):
<a name="cl-2440"></a>                for global_col in xrange(global_block_col*block_col_size, min((global_block_col+1)*block_col_size, n_cols)):
<a name="cl-2441"></a>                    #is cache block not loaded?
<a name="cl-2442"></a>
<a name="cl-2443"></a>                    e_0_row = e_0_offsets[0] + global_row
<a name="cl-2444"></a>                    e_0_col = e_0_offsets[1] + global_col
<a name="cl-2445"></a>
<a name="cl-2446"></a>                    block_cache.update_cache(e_0_row, e_0_col, &amp;e_0_row_index, &amp;e_0_col_index, &amp;e_0_row_block_offset, &amp;e_0_col_block_offset)
<a name="cl-2447"></a>
<a name="cl-2448"></a>                    current_label = label_block[
<a name="cl-2449"></a>                        e_0_row_index, e_0_col_index,
<a name="cl-2450"></a>                        e_0_row_block_offset, e_0_col_block_offset]
<a name="cl-2451"></a>
<a name="cl-2452"></a>                    #if a label isn't defiend we're not in a flat region
<a name="cl-2453"></a>                    if current_label == label_nodata:
<a name="cl-2454"></a>                        continue
<a name="cl-2455"></a>
<a name="cl-2456"></a>                    current_flow = flow_block[
<a name="cl-2457"></a>                        e_0_row_index, e_0_col_index,
<a name="cl-2458"></a>                        e_0_row_block_offset, e_0_col_block_offset]
<a name="cl-2459"></a>
<a name="cl-2460"></a>                    #this can happen if we have been passed an existing flow
<a name="cl-2461"></a>                    #direction raster, perhaps from an earlier iteration in a
<a name="cl-2462"></a>                    #multiphase flow resolution algorithm
<a name="cl-2463"></a>                    if current_flow != flow_nodata:
<a name="cl-2464"></a>                        continue
<a name="cl-2465"></a>
<a name="cl-2466"></a>                    e_0 = flat_mask_block[e_0_row_index, e_0_col_index, e_0_row_block_offset, e_0_col_block_offset]
<a name="cl-2467"></a>                    #skip if we're on a nodata pixel skip
<a name="cl-2468"></a>
<a name="cl-2469"></a>                    #Calculate the flow flow_direction for each facet
<a name="cl-2470"></a>                    slope_max = 0 #use this to keep track of the maximum down-slope
<a name="cl-2471"></a>                    flow_direction_max_slope = 0 #flow direction on max downward slope
<a name="cl-2472"></a>                    max_index = 0 #index to keep track of max slope facet
<a name="cl-2473"></a>
<a name="cl-2474"></a>                    for facet_index in range(8):
<a name="cl-2475"></a>                        #This defines the three points the facet
<a name="cl-2476"></a>
<a name="cl-2477"></a>                        e_1_row = e_1_offsets[facet_index * 2 + 0] + global_row
<a name="cl-2478"></a>                        e_1_col = e_1_offsets[facet_index * 2 + 1] + global_col
<a name="cl-2479"></a>                        e_2_row = e_2_offsets[facet_index * 2 + 0] + global_row
<a name="cl-2480"></a>                        e_2_col = e_2_offsets[facet_index * 2 + 1] + global_col
<a name="cl-2481"></a>                        #make sure one of the facets doesn't hang off the edge
<a name="cl-2482"></a>                        if (e_1_row &lt; 0 or e_1_row &gt;= n_rows or
<a name="cl-2483"></a>                            e_2_row &lt; 0 or e_2_row &gt;= n_rows or
<a name="cl-2484"></a>                            e_1_col &lt; 0 or e_1_col &gt;= n_cols or
<a name="cl-2485"></a>                            e_2_col &lt; 0 or e_2_col &gt;= n_cols):
<a name="cl-2486"></a>                            continue
<a name="cl-2487"></a>
<a name="cl-2488"></a>                        block_cache.update_cache(e_1_row, e_1_col, &amp;e_1_row_index, &amp;e_1_col_index, &amp;e_1_row_block_offset, &amp;e_1_col_block_offset)
<a name="cl-2489"></a>                        block_cache.update_cache(e_2_row, e_2_col, &amp;e_2_row_index, &amp;e_2_col_index, &amp;e_2_row_block_offset, &amp;e_2_col_block_offset)
<a name="cl-2490"></a>
<a name="cl-2491"></a>                        e_1 = flat_mask_block[e_1_row_index, e_1_col_index, e_1_row_block_offset, e_1_col_block_offset]
<a name="cl-2492"></a>                        e_2 = flat_mask_block[e_2_row_index, e_2_col_index, e_2_row_block_offset, e_2_col_block_offset]
<a name="cl-2493"></a>
<a name="cl-2494"></a>                        e_1_label = label_block[e_1_row_index, e_1_col_index, e_1_row_block_offset, e_1_col_block_offset]
<a name="cl-2495"></a>                        e_2_label = label_block[e_2_row_index, e_2_col_index, e_2_row_block_offset, e_2_col_block_offset]
<a name="cl-2496"></a>
<a name="cl-2497"></a>                        #if labels aren't t the same as the current, we can't flow to them
<a name="cl-2498"></a>                        if e_1_label != current_label and e_2_label != current_label:
<a name="cl-2499"></a>                            continue
<a name="cl-2500"></a>
<a name="cl-2501"></a>                        #s_1 is slope along straight edge
<a name="cl-2502"></a>                        s_1 = (e_0 - e_1) / d_1 #Eqn 1
<a name="cl-2503"></a>                        #slope along diagonal edge
<a name="cl-2504"></a>                        s_2 = (e_1 - e_2) / d_2 #Eqn 2
<a name="cl-2505"></a>
<a name="cl-2506"></a>                        #can't calculate flow direction if one of the facets is nodata
<a name="cl-2507"></a>                        if e_1_label != current_label or e_2_label != current_label:
<a name="cl-2508"></a>                            #make sure the flow direction perfectly aligns with
<a name="cl-2509"></a>                            #the facet direction so we don't get a case where
<a name="cl-2510"></a>                            #we point toward a pixel but the next pixel down
<a name="cl-2511"></a>                            #is the correct flow direction
<a name="cl-2512"></a>                            if e_1_label == current_label and facet_index % 2 == 0 and e_1 &lt; e_0:
<a name="cl-2513"></a>                                #straight line to next pixel
<a name="cl-2514"></a>                                slope = s_1
<a name="cl-2515"></a>                                flow_direction = 0
<a name="cl-2516"></a>                            elif e_2_label == current_label and facet_index % 2 == 1 and e_2 &lt; e_0:
<a name="cl-2517"></a>                                #diagonal line to next pixel
<a name="cl-2518"></a>                                slope = (e_0 - e_2) / sqrt(d_1 **2 + d_2 ** 2)
<a name="cl-2519"></a>                                flow_direction = max_r
<a name="cl-2520"></a>                            else:
<a name="cl-2521"></a>                                continue
<a name="cl-2522"></a>                        else:
<a name="cl-2523"></a>                            #both facets are defined, this is the core of
<a name="cl-2524"></a>                            #d-infinity algorithm
<a name="cl-2525"></a>                            flow_direction = atan2(s_2, s_1) #Eqn 3
<a name="cl-2526"></a>
<a name="cl-2527"></a>                            if flow_direction &lt; 0: #Eqn 4
<a name="cl-2528"></a>                                #If the flow direction goes off one side, set flow
<a name="cl-2529"></a>                                #direction to that side and the slope to the straight line
<a name="cl-2530"></a>                                #distance slope
<a name="cl-2531"></a>                                flow_direction = 0
<a name="cl-2532"></a>                                slope = s_1
<a name="cl-2533"></a>                            elif flow_direction &gt; max_r: #Eqn 5
<a name="cl-2534"></a>                                #If the flow direciton goes off the diagonal side, figure
<a name="cl-2535"></a>                                #out what its value is and
<a name="cl-2536"></a>                                flow_direction = max_r
<a name="cl-2537"></a>                                slope = (e_0 - e_2) / sqrt(d_1 ** 2 + d_2 ** 2)
<a name="cl-2538"></a>                            else:
<a name="cl-2539"></a>                                slope = sqrt(s_1 ** 2 + s_2 ** 2) #Eqn 3
<a name="cl-2540"></a>
<a name="cl-2541"></a>                        #update the maxes depending on the results above
<a name="cl-2542"></a>                        if slope &gt; slope_max:
<a name="cl-2543"></a>                            flow_direction_max_slope = flow_direction
<a name="cl-2544"></a>                            slope_max = slope
<a name="cl-2545"></a>                            max_index = facet_index
<a name="cl-2546"></a>
<a name="cl-2547"></a>                    #if there's a downward slope, save the flow direction
<a name="cl-2548"></a>                    if slope_max &gt; 0:
<a name="cl-2549"></a>                        flow_block[e_0_row_index, e_0_col_index, e_0_row_block_offset, e_0_col_block_offset] = (
<a name="cl-2550"></a>                            a_f[max_index] * flow_direction_max_slope +
<a name="cl-2551"></a>                            a_c[max_index] * PI / 2.0)
<a name="cl-2552"></a>                        cache_dirty[e_0_row_index, e_0_col_index] = 1
<a name="cl-2553"></a>
<a name="cl-2554"></a>    block_cache.flush_cache()
<a name="cl-2555"></a>    flow_band = None
<a name="cl-2556"></a>    gdal.Dataset.__swig_destroy__(flow_direction_dataset)
<a name="cl-2557"></a>    flow_direction_dataset = None
<a name="cl-2558"></a>    pygeoprocessing.calculate_raster_stats_uri(flow_direction_uri)
<a name="cl-2559"></a>
<a name="cl-2560"></a>
<a name="cl-2561"></a>#@cython.boundscheck(False)
<a name="cl-2562"></a>@cython.wraparound(False)
<a name="cl-2563"></a>@cython.cdivision(True)
<a name="cl-2564"></a>cdef find_outlets(dem_uri, flow_direction_uri, deque[int] &amp;outlet_deque):
<a name="cl-2565"></a>    """Discover and return the outlets in the dem array
<a name="cl-2566"></a>
<a name="cl-2567"></a>        Args:
<a name="cl-2568"></a>            dem_uri (string) - (input) a uri to a gdal dataset representing
<a name="cl-2569"></a>                height values
<a name="cl-2570"></a>            flow_direction_uri (string) - (input) a uri to gdal dataset
<a name="cl-2571"></a>                representing flow direction values
<a name="cl-2572"></a>            outlet_deque (deque[int]) - (output) a reference to a c++ set that
<a name="cl-2573"></a>                contains the set of flat integer index indicating the outlets
<a name="cl-2574"></a>                in dem
<a name="cl-2575"></a>
<a name="cl-2576"></a>        Returns:
<a name="cl-2577"></a>            nothing"""
<a name="cl-2578"></a>
<a name="cl-2579"></a>    dem_ds = gdal.Open(dem_uri)
<a name="cl-2580"></a>    dem_band = dem_ds.GetRasterBand(1)
<a name="cl-2581"></a>
<a name="cl-2582"></a>    flow_direction_ds = gdal.Open(flow_direction_uri)
<a name="cl-2583"></a>    flow_direction_band = flow_direction_ds.GetRasterBand(1)
<a name="cl-2584"></a>    cdef float flow_nodata = pygeoprocessing.get_nodata_from_uri(
<a name="cl-2585"></a>        flow_direction_uri)
<a name="cl-2586"></a>
<a name="cl-2587"></a>    cdef int block_col_size, block_row_size
<a name="cl-2588"></a>    block_col_size, block_row_size = dem_band.GetBlockSize()
<a name="cl-2589"></a>    cdef int n_rows = dem_ds.RasterYSize
<a name="cl-2590"></a>    cdef int n_cols = dem_ds.RasterXSize
<a name="cl-2591"></a>
<a name="cl-2592"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] dem_block = numpy.zeros(
<a name="cl-2593"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
<a name="cl-2594"></a>        dtype=numpy.float32)
<a name="cl-2595"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] flow_direction_block = numpy.zeros(
<a name="cl-2596"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
<a name="cl-2597"></a>        dtype=numpy.float32)
<a name="cl-2598"></a>
<a name="cl-2599"></a>    band_list = [dem_band, flow_direction_band]
<a name="cl-2600"></a>    block_list = [dem_block, flow_direction_block]
<a name="cl-2601"></a>    update_list = [False, False]
<a name="cl-2602"></a>    cdef numpy.ndarray[numpy.npy_byte, ndim=2] cache_dirty = numpy.zeros(
<a name="cl-2603"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.byte)
<a name="cl-2604"></a>
<a name="cl-2605"></a>    cdef BlockCache_SWY block_cache = BlockCache_SWY(
<a name="cl-2606"></a>        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size,
<a name="cl-2607"></a>        block_col_size, band_list, block_list, update_list, cache_dirty)
<a name="cl-2608"></a>
<a name="cl-2609"></a>    cdef float dem_nodata = pygeoprocessing.get_nodata_from_uri(dem_uri)
<a name="cl-2610"></a>
<a name="cl-2611"></a>    cdef int cell_row_index, cell_col_index
<a name="cl-2612"></a>    cdef int cell_row_block_index, cell_col_block_index
<a name="cl-2613"></a>    cdef int cell_row_block_offset, cell_col_block_offset
<a name="cl-2614"></a>    cdef int flat_index
<a name="cl-2615"></a>    cdef float dem_value, flow_direction
<a name="cl-2616"></a>
<a name="cl-2617"></a>    outlet_deque.clear()
<a name="cl-2618"></a>
<a name="cl-2619"></a>    cdef time_t last_time, current_time
<a name="cl-2620"></a>    time(&amp;last_time)
<a name="cl-2621"></a>
<a name="cl-2622"></a>    for cell_row_index in xrange(n_rows):
<a name="cl-2623"></a>        time(&amp;current_time)
<a name="cl-2624"></a>        if current_time - last_time &gt; 5.0:
<a name="cl-2625"></a>            LOGGER.info(
<a name="cl-2626"></a>                'find outlet percent complete = %.2f, outlet_deque size = %d',
<a name="cl-2627"></a>                float(cell_row_index)/n_rows * 100, outlet_deque.size())
<a name="cl-2628"></a>            last_time = current_time
<a name="cl-2629"></a>        for cell_col_index in xrange(n_cols):
<a name="cl-2630"></a>
<a name="cl-2631"></a>            block_cache.update_cache(
<a name="cl-2632"></a>                cell_row_index, cell_col_index,
<a name="cl-2633"></a>                &amp;cell_row_block_index, &amp;cell_col_block_index,
<a name="cl-2634"></a>                &amp;cell_row_block_offset, &amp;cell_col_block_offset)
<a name="cl-2635"></a>
<a name="cl-2636"></a>            dem_value = dem_block[
<a name="cl-2637"></a>                cell_row_block_index, cell_col_block_index,
<a name="cl-2638"></a>                cell_row_block_offset, cell_col_block_offset]
<a name="cl-2639"></a>            flow_direction = flow_direction_block[
<a name="cl-2640"></a>                cell_row_block_index, cell_col_block_index,
<a name="cl-2641"></a>                cell_row_block_offset, cell_col_block_offset]
<a name="cl-2642"></a>
<a name="cl-2643"></a>            #it's a valid dem but no flow direction could be defined, it's
<a name="cl-2644"></a>            #either a sink or an outlet
<a name="cl-2645"></a>
<a name="cl-2646"></a>            if dem_value != dem_nodata and flow_direction == flow_nodata:
<a name="cl-2647"></a>                flat_index = cell_row_index * n_cols + cell_col_index
<a name="cl-2648"></a>                outlet_deque.push_front(flat_index)
<a name="cl-2649"></a>
<a name="cl-2650"></a>
<a name="cl-2651"></a>def resolve_flats(
<a name="cl-2652"></a>    dem_uri, flow_direction_uri, flat_mask_uri, labels_uri,
<a name="cl-2653"></a>    drain_off_edge=False):
<a name="cl-2654"></a>    """Function to resolve the flat regions in the dem given a first attempt
<a name="cl-2655"></a>        run at calculating flow direction.  Will provide regions of flat areas
<a name="cl-2656"></a>        and their labels.
<a name="cl-2657"></a>
<a name="cl-2658"></a>        Based on: Barnes, Richard, Clarence Lehman, and David Mulla. "An
<a name="cl-2659"></a>            efficient assignment of drainage direction over flat surfaces in
<a name="cl-2660"></a>            raster digital elevation models." Computers &amp; Geosciences 62
<a name="cl-2661"></a>            (2014): 128-135.
<a name="cl-2662"></a>
<a name="cl-2663"></a>        Args:
<a name="cl-2664"></a>            dem_uri (string) - (input) a uri to a single band GDAL Dataset with
<a name="cl-2665"></a>                elevation values
<a name="cl-2666"></a>            flow_direction_uri (string) - (input/output) a uri to a single band
<a name="cl-2667"></a>                GDAL Dataset with partially defined d_infinity flow directions
<a name="cl-2668"></a>            drain_off_edge (boolean) - input if true will drain flat areas off
<a name="cl-2669"></a>                the edge of the raster
<a name="cl-2670"></a>
<a name="cl-2671"></a>        Returns:
<a name="cl-2672"></a>            True if there were flats to resolve, False otherwise"""
<a name="cl-2673"></a>
<a name="cl-2674"></a>    cdef deque[int] high_edges
<a name="cl-2675"></a>    cdef deque[int] low_edges
<a name="cl-2676"></a>    flat_edges(
<a name="cl-2677"></a>        dem_uri, flow_direction_uri, high_edges, low_edges,
<a name="cl-2678"></a>        drain_off_edge=drain_off_edge)
<a name="cl-2679"></a>
<a name="cl-2680"></a>    if low_edges.size() == 0:
<a name="cl-2681"></a>        if high_edges.size() != 0:
<a name="cl-2682"></a>            LOGGER.warn('There were undrainable flats')
<a name="cl-2683"></a>        else:
<a name="cl-2684"></a>            LOGGER.info('There were no flats')
<a name="cl-2685"></a>        return False
<a name="cl-2686"></a>
<a name="cl-2687"></a>    LOGGER.info('labeling flats')
<a name="cl-2688"></a>    label_flats(dem_uri, low_edges, labels_uri)
<a name="cl-2689"></a>
<a name="cl-2690"></a>    drain_flats(
<a name="cl-2691"></a>        high_edges, low_edges, labels_uri, flow_direction_uri, flat_mask_uri)
<a name="cl-2692"></a>
<a name="cl-2693"></a>    return True
<a name="cl-2694"></a>
<a name="cl-2695"></a>
<a name="cl-2696"></a>def calculate_recharge(
<a name="cl-2697"></a>    precip_uri_list, et0_uri_list, qfi_uri_list, flow_dir_uri, outflow_weights_uri,
<a name="cl-2698"></a>    outflow_direction_uri, dem_uri, lulc_uri, kc_lookup, alpha_m, beta_i, gamma,
<a name="cl-2699"></a>    stream_uri, recharge_uri, recharge_avail_uri, r_sum_avail_uri,
<a name="cl-2700"></a>    aet_uri, kc_uri):
<a name="cl-2701"></a>
<a name="cl-2702"></a>    cdef deque[int] outlet_cell_deque
<a name="cl-2703"></a>    find_outlets(
<a name="cl-2704"></a>        dem_uri, flow_dir_uri, outlet_cell_deque)
<a name="cl-2705"></a>    route_recharge(
<a name="cl-2706"></a>        precip_uri_list, et0_uri_list, kc_uri, recharge_uri, recharge_avail_uri,
<a name="cl-2707"></a>        r_sum_avail_uri, aet_uri, alpha_m, beta_i, gamma, qfi_uri_list,
<a name="cl-2708"></a>        outflow_direction_uri, outflow_weights_uri, stream_uri,
<a name="cl-2709"></a>        outlet_cell_deque)
<a name="cl-2710"></a>
<a name="cl-2711"></a>
<a name="cl-2712"></a>def calculate_r_sum_avail_pour(
<a name="cl-2713"></a>        r_sum_avail_uri, outflow_weights_uri, outflow_direction_uri,
<a name="cl-2714"></a>        r_sum_avail_pour_uri):
<a name="cl-2715"></a>    """Calculate how r_sum_avail r_sum_avail_pours directly into its neighbors"""
<a name="cl-2716"></a>
<a name="cl-2717"></a>    out_dir = os.path.dirname(r_sum_avail_uri)
<a name="cl-2718"></a>    r_sum_avail_ds = gdal.Open(r_sum_avail_uri)
<a name="cl-2719"></a>    r_sum_avail_band = r_sum_avail_ds.GetRasterBand(1)
<a name="cl-2720"></a>    block_col_size, block_row_size = r_sum_avail_band.GetBlockSize()
<a name="cl-2721"></a>    r_sum_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(
<a name="cl-2722"></a>        r_sum_avail_uri)
<a name="cl-2723"></a>
<a name="cl-2724"></a>    cdef float r_sum_avail_pour_nodata = -1.0
<a name="cl-2725"></a>    pygeoprocessing.new_raster_from_base_uri(
<a name="cl-2726"></a>        r_sum_avail_uri, r_sum_avail_pour_uri, 'GTiff', r_sum_avail_pour_nodata,
<a name="cl-2727"></a>        gdal.GDT_Float32)
<a name="cl-2728"></a>    r_sum_avail_pour_dataset = gdal.Open(r_sum_avail_pour_uri, gdal.GA_Update)
<a name="cl-2729"></a>    r_sum_avail_pour_band = r_sum_avail_pour_dataset.GetRasterBand(1)
<a name="cl-2730"></a>
<a name="cl-2731"></a>    n_rows = r_sum_avail_band.YSize
<a name="cl-2732"></a>    n_cols = r_sum_avail_band.XSize
<a name="cl-2733"></a>
<a name="cl-2734"></a>    n_global_block_rows = int(numpy.ceil(float(n_rows) / block_row_size))
<a name="cl-2735"></a>    n_global_block_cols = int(numpy.ceil(float(n_cols) / block_col_size))
<a name="cl-2736"></a>
<a name="cl-2737"></a>    cdef numpy.ndarray[numpy.npy_int8, ndim=4] outflow_direction_block = numpy.zeros(
<a name="cl-2738"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.int8)
<a name="cl-2739"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] outflow_weights_block = numpy.zeros(
<a name="cl-2740"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-2741"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] r_sum_avail_block = numpy.zeros(
<a name="cl-2742"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-2743"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] r_sum_avail_pour_block = numpy.zeros(
<a name="cl-2744"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-2745"></a>
<a name="cl-2746"></a>    cdef numpy.ndarray[numpy.npy_int8, ndim=2] cache_dirty = numpy.zeros(
<a name="cl-2747"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.int8)
<a name="cl-2748"></a>
<a name="cl-2749"></a>    outflow_direction_dataset = gdal.Open(outflow_direction_uri)
<a name="cl-2750"></a>    outflow_direction_band = outflow_direction_dataset.GetRasterBand(1)
<a name="cl-2751"></a>    cdef float outflow_direction_nodata = pygeoprocessing.get_nodata_from_uri(
<a name="cl-2752"></a>        outflow_direction_uri)
<a name="cl-2753"></a>    outflow_weights_dataset = gdal.Open(outflow_weights_uri)
<a name="cl-2754"></a>    outflow_weights_band = outflow_weights_dataset.GetRasterBand(1)
<a name="cl-2755"></a>    cdef float outflow_weights_nodata = pygeoprocessing.get_nodata_from_uri(
<a name="cl-2756"></a>        outflow_weights_uri)
<a name="cl-2757"></a>
<a name="cl-2758"></a>    #make the memory block
<a name="cl-2759"></a>    band_list = [
<a name="cl-2760"></a>        r_sum_avail_band, outflow_direction_band, outflow_weights_band,
<a name="cl-2761"></a>        r_sum_avail_pour_band]
<a name="cl-2762"></a>    block_list = [
<a name="cl-2763"></a>        r_sum_avail_block, outflow_direction_block, outflow_weights_block,
<a name="cl-2764"></a>        r_sum_avail_pour_block]
<a name="cl-2765"></a>
<a name="cl-2766"></a>    update_list = [False, False, False, True]
<a name="cl-2767"></a>    cache_dirty[:] = 0
<a name="cl-2768"></a>
<a name="cl-2769"></a>    cdef BlockCache_SWY block_cache = BlockCache_SWY(
<a name="cl-2770"></a>        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols,
<a name="cl-2771"></a>        block_row_size, block_col_size,
<a name="cl-2772"></a>        band_list, block_list, update_list, cache_dirty)
<a name="cl-2773"></a>
<a name="cl-2774"></a>    #center point of global index
<a name="cl-2775"></a>    cdef int global_row, global_col #index into the overall raster
<a name="cl-2776"></a>    cdef int row_index, col_index #the index of the cache block
<a name="cl-2777"></a>    cdef int row_block_offset, col_block_offset #index into the cache block
<a name="cl-2778"></a>    cdef int global_block_row, global_block_col #used to walk the global blocks
<a name="cl-2779"></a>
<a name="cl-2780"></a>    #neighbor sections of global index
<a name="cl-2781"></a>    cdef int neighbor_row, neighbor_col #neighbor equivalent of global_{row,col}
<a name="cl-2782"></a>    cdef int neighbor_row_index, neighbor_col_index #neighbor cache index
<a name="cl-2783"></a>    cdef int neighbor_row_block_offset, neighbor_col_block_offset #index into the neighbor cache block
<a name="cl-2784"></a>
<a name="cl-2785"></a>    cdef int *row_offsets = [0, -1, -1, -1,  0,  1, 1, 1]
<a name="cl-2786"></a>    cdef int *col_offsets = [1,  1,  0, -1, -1, -1, 0, 1]
<a name="cl-2787"></a>    cdef int *inflow_offsets = [4, 5, 6, 7, 0, 1, 2, 3]
<a name="cl-2788"></a>
<a name="cl-2789"></a>    for global_block_row in xrange(n_global_block_rows):
<a name="cl-2790"></a>        for global_block_col in xrange(n_global_block_cols):
<a name="cl-2791"></a>            xoff = global_block_col * block_col_size
<a name="cl-2792"></a>            yoff = global_block_row * block_row_size
<a name="cl-2793"></a>            win_xsize = min(block_col_size, n_cols - xoff)
<a name="cl-2794"></a>            win_ysize = min(block_row_size, n_rows - yoff)
<a name="cl-2795"></a>
<a name="cl-2796"></a>            for global_row in xrange(yoff, yoff+win_ysize):
<a name="cl-2797"></a>                for global_col in xrange(xoff, xoff+win_xsize):
<a name="cl-2798"></a>
<a name="cl-2799"></a>                    block_cache.update_cache(global_row, global_col, &amp;row_index, &amp;col_index, &amp;row_block_offset, &amp;col_block_offset)
<a name="cl-2800"></a>                    if r_sum_avail_block[row_index, col_index, row_block_offset, col_block_offset] == r_sum_nodata:
<a name="cl-2801"></a>                        r_sum_avail_pour_block[row_index, col_index, row_block_offset, col_block_offset] = r_sum_avail_pour_nodata
<a name="cl-2802"></a>                        cache_dirty[row_index, col_index] = 1
<a name="cl-2803"></a>                        continue
<a name="cl-2804"></a>
<a name="cl-2805"></a>                    r_sum_avail_pour_sum = 0.0
<a name="cl-2806"></a>                    for direction_index in xrange(8):
<a name="cl-2807"></a>                        #get percent flow from neighbor to current cell
<a name="cl-2808"></a>                        neighbor_row = global_row + row_offsets[direction_index]
<a name="cl-2809"></a>                        neighbor_col = global_col + col_offsets[direction_index]
<a name="cl-2810"></a>
<a name="cl-2811"></a>                        #See if neighbor out of bounds
<a name="cl-2812"></a>                        if (neighbor_row &lt; 0 or neighbor_row &gt;= n_rows or neighbor_col &lt; 0 or neighbor_col &gt;= n_cols):
<a name="cl-2813"></a>                            continue
<a name="cl-2814"></a>
<a name="cl-2815"></a>                        block_cache.update_cache(neighbor_row, neighbor_col, &amp;neighbor_row_index, &amp;neighbor_col_index, &amp;neighbor_row_block_offset, &amp;neighbor_col_block_offset)
<a name="cl-2816"></a>                        #if neighbor inflows
<a name="cl-2817"></a>                        neighbor_direction = outflow_direction_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset]
<a name="cl-2818"></a>                        if neighbor_direction == outflow_direction_nodata:
<a name="cl-2819"></a>                            continue
<a name="cl-2820"></a>
<a name="cl-2821"></a>                        if r_sum_avail_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset] == r_sum_nodata:
<a name="cl-2822"></a>                            continue
<a name="cl-2823"></a>
<a name="cl-2824"></a>                        #check if the cell flows directly, or is one index off
<a name="cl-2825"></a>                        if (inflow_offsets[direction_index] != neighbor_direction and
<a name="cl-2826"></a>                                ((inflow_offsets[direction_index] - 1) % 8) != neighbor_direction):
<a name="cl-2827"></a>                            #then neighbor doesn't inflow into current cell
<a name="cl-2828"></a>                            continue
<a name="cl-2829"></a>
<a name="cl-2830"></a>                        #Calculate the outflow weight
<a name="cl-2831"></a>                        outflow_weight = outflow_weights_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset]
<a name="cl-2832"></a>
<a name="cl-2833"></a>                        if ((inflow_offsets[direction_index] - 1) % 8) == neighbor_direction:
<a name="cl-2834"></a>                            outflow_weight = 1.0 - outflow_weight
<a name="cl-2835"></a>
<a name="cl-2836"></a>                        if outflow_weight &lt;= 0.0:
<a name="cl-2837"></a>                            continue
<a name="cl-2838"></a>                        r_sum_avail_pour_sum += r_sum_avail_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset] * outflow_weight
<a name="cl-2839"></a>
<a name="cl-2840"></a>                    block_cache.update_cache(global_row, global_col, &amp;row_index, &amp;col_index, &amp;row_block_offset, &amp;col_block_offset)
<a name="cl-2841"></a>                    r_sum_avail_pour_block[row_index, col_index, row_block_offset, col_block_offset] = r_sum_avail_pour_sum
<a name="cl-2842"></a>                    cache_dirty[row_index, col_index] = 1
<a name="cl-2843"></a>    block_cache.flush_cache()
<a name="cl-2844"></a>
<a name="cl-2845"></a>
<a name="cl-2846"></a>@cython.wraparound(False)
<a name="cl-2847"></a>@cython.cdivision(True)
<a name="cl-2848"></a>def route_sf(
<a name="cl-2849"></a>    dem_uri, r_avail_uri, r_sum_avail_uri, r_sum_avail_pour_uri,
<a name="cl-2850"></a>    outflow_direction_uri, outflow_weights_uri, stream_uri, sf_uri,
<a name="cl-2851"></a>    sf_down_uri):
<a name="cl-2852"></a>
<a name="cl-2853"></a>    #Pass transport
<a name="cl-2854"></a>    cdef time_t start
<a name="cl-2855"></a>    time(&amp;start)
<a name="cl-2856"></a>
<a name="cl-2857"></a>    cdef deque[int] cells_to_process
<a name="cl-2858"></a>    find_outlets(dem_uri, outflow_direction_uri, cells_to_process)
<a name="cl-2859"></a>
<a name="cl-2860"></a>    cdef c_set[int] cells_in_queue
<a name="cl-2861"></a>    for cell in cells_to_process:
<a name="cl-2862"></a>        cells_in_queue.insert(cell)
<a name="cl-2863"></a>
<a name="cl-2864"></a>    cdef float pixel_area = (
<a name="cl-2865"></a>        pygeoprocessing.geoprocessing.get_cell_size_from_uri(dem_uri) ** 2)
<a name="cl-2866"></a>
<a name="cl-2867"></a>    #load a base dataset so we can determine the n_rows/cols
<a name="cl-2868"></a>    outflow_direction_dataset = gdal.Open(outflow_direction_uri, gdal.GA_ReadOnly)
<a name="cl-2869"></a>    cdef int n_cols = outflow_direction_dataset.RasterXSize
<a name="cl-2870"></a>    cdef int n_rows = outflow_direction_dataset.RasterYSize
<a name="cl-2871"></a>    outflow_direction_band = outflow_direction_dataset.GetRasterBand(1)
<a name="cl-2872"></a>
<a name="cl-2873"></a>    cdef int block_col_size, block_row_size
<a name="cl-2874"></a>    block_col_size, block_row_size = outflow_direction_band.GetBlockSize()
<a name="cl-2875"></a>
<a name="cl-2876"></a>    #center point of global index
<a name="cl-2877"></a>    cdef int global_row, global_col #index into the overall raster
<a name="cl-2878"></a>    cdef int row_index, col_index #the index of the cache block
<a name="cl-2879"></a>    cdef int row_block_offset, col_block_offset #index into the cache block
<a name="cl-2880"></a>    cdef int global_block_row, global_block_col #used to walk the global blocks
<a name="cl-2881"></a>
<a name="cl-2882"></a>    #neighbor sections of global index
<a name="cl-2883"></a>    cdef int neighbor_row, neighbor_col #neighbor equivalent of global_{row,col}
<a name="cl-2884"></a>    cdef int neighbor_row_index, neighbor_col_index #neighbor cache index
<a name="cl-2885"></a>    cdef int neighbor_row_block_offset, neighbor_col_block_offset #index into the neighbor cache block
<a name="cl-2886"></a>
<a name="cl-2887"></a>    #define all the single caches
<a name="cl-2888"></a>    cdef numpy.ndarray[numpy.npy_int8, ndim=4] outflow_direction_block = numpy.zeros(
<a name="cl-2889"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.int8)
<a name="cl-2890"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] outflow_weights_block = numpy.zeros(
<a name="cl-2891"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-2892"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] r_avail_block = numpy.zeros(
<a name="cl-2893"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-2894"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] r_sum_avail_block = numpy.zeros(
<a name="cl-2895"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-2896"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] r_sum_avail_pour_block = numpy.zeros(
<a name="cl-2897"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-2898"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] sf_down_block = numpy.zeros(
<a name="cl-2899"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-2900"></a>    cdef numpy.ndarray[numpy.npy_float32, ndim=4] sf_block = numpy.zeros(
<a name="cl-2901"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
<a name="cl-2902"></a>    cdef numpy.ndarray[numpy.npy_int8, ndim=4] stream_block = numpy.zeros(
<a name="cl-2903"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.int8)
<a name="cl-2904"></a>
<a name="cl-2905"></a>    cdef numpy.ndarray[numpy.npy_int8, ndim=2] cache_dirty = numpy.zeros(
<a name="cl-2906"></a>        (N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.int8)
<a name="cl-2907"></a>
<a name="cl-2908"></a>    cdef int outflow_direction_nodata = pygeoprocessing.get_nodata_from_uri(
<a name="cl-2909"></a>        outflow_direction_uri)
<a name="cl-2910"></a>
<a name="cl-2911"></a>    outflow_weights_dataset = gdal.Open(outflow_weights_uri)
<a name="cl-2912"></a>    outflow_weights_band = outflow_weights_dataset.GetRasterBand(1)
<a name="cl-2913"></a>    cdef float outflow_weights_nodata = pygeoprocessing.get_nodata_from_uri(
<a name="cl-2914"></a>        outflow_weights_uri)
<a name="cl-2915"></a>
<a name="cl-2916"></a>    #Create output arrays qfi and recharge and recharge_avail
<a name="cl-2917"></a>    r_avail_dataset = gdal.Open(r_avail_uri)
<a name="cl-2918"></a>    r_avail_band = r_avail_dataset.GetRasterBand(1)
<a name="cl-2919"></a>
<a name="cl-2920"></a>    r_sum_avail_dataset = gdal.Open(r_sum_avail_uri)
<a name="cl-2921"></a>    r_sum_avail_band = r_sum_avail_dataset.GetRasterBand(1)
<a name="cl-2922"></a>    cdef float r_sum_nodata = r_sum_avail_band.GetNoDataValue()
<a name="cl-2923"></a>
<a name="cl-2924"></a>    r_sum_avail_pour_dataset = gdal.Open(r_sum_avail_pour_uri)
<a name="cl-2925"></a>    r_sum_avail_pour_band = r_sum_avail_pour_dataset.GetRasterBand(1)
<a name="cl-2926"></a>
<a name="cl-2927"></a>    stream_dataset = gdal.Open(stream_uri, gdal.GA_ReadOnly)
<a name="cl-2928"></a>    stream_band = stream_dataset.GetRasterBand(1)
<a name="cl-2929"></a>
<a name="cl-2930"></a>    cdef float sf_down_nodata = -9999.0
<a name="cl-2931"></a>    pygeoprocessing.new_raster_from_base_uri(
<a name="cl-2932"></a>        outflow_direction_uri, sf_down_uri, 'GTiff', sf_down_nodata,
<a name="cl-2933"></a>        gdal.GDT_Float32, fill_value=sf_down_nodata)
<a name="cl-2934"></a>    sf_down_dataset = gdal.Open(sf_down_uri, gdal.GA_Update)
<a name="cl-2935"></a>    sf_down_band = sf_down_dataset.GetRasterBand(1)
<a name="cl-2936"></a>
<a name="cl-2937"></a>    cdef float sf_nodata = -9999.0
<a name="cl-2938"></a>    pygeoprocessing.new_raster_from_base_uri(
<a name="cl-2939"></a>        outflow_direction_uri, sf_uri, 'GTiff', sf_nodata,
<a name="cl-2940"></a>        gdal.GDT_Float32, fill_value=sf_nodata)
<a name="cl-2941"></a>    sf_dataset = gdal.Open(sf_uri, gdal.GA_Update)
<a name="cl-2942"></a>    sf_band = sf_dataset.GetRasterBand(1)
<a name="cl-2943"></a>
<a name="cl-2944"></a>
<a name="cl-2945"></a>    band_list = [
<a name="cl-2946"></a>        outflow_direction_band, outflow_weights_band, r_avail_band, r_sum_avail_band,
<a name="cl-2947"></a>        r_sum_avail_pour_band, stream_band, sf_down_band, sf_band]
<a name="cl-2948"></a>    block_list = [
<a name="cl-2949"></a>        outflow_direction_block, outflow_weights_block, r_avail_block, r_sum_avail_block,
<a name="cl-2950"></a>        r_sum_avail_pour_block, stream_block, sf_down_block, sf_block]
<a name="cl-2951"></a>    update_list = [False] * 6 + [True] * 2
<a name="cl-2952"></a>    cache_dirty[:] = 0
<a name="cl-2953"></a>
<a name="cl-2954"></a>    cdef BlockCache_SWY block_cache = BlockCache_SWY(
<a name="cl-2955"></a>        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols,
<a name="cl-2956"></a>        block_row_size, block_col_size,
<a name="cl-2957"></a>        band_list, block_list, update_list, cache_dirty)
<a name="cl-2958"></a>
<a name="cl-2959"></a>    #Diagonal offsets are based off the following index notation for neighbors
<a name="cl-2960"></a>    #    3 2 1
<a name="cl-2961"></a>    #    4 p 0
<a name="cl-2962"></a>    #    5 6 7
<a name="cl-2963"></a>
<a name="cl-2964"></a>    cdef int *row_offsets = [0, -1, -1, -1,  0,  1, 1, 1]
<a name="cl-2965"></a>    cdef int *col_offsets = [1,  1,  0, -1, -1, -1, 0, 1]
<a name="cl-2966"></a>    cdef int *neighbor_row_offset = [0, -1, -1, -1,  0,  1, 1, 1]
<a name="cl-2967"></a>    cdef int *neighbor_col_offset = [1,  1,  0, -1, -1, -1, 0, 1]
<a name="cl-2968"></a>    cdef int *inflow_offsets = [4, 5, 6, 7, 0, 1, 2, 3]
<a name="cl-2969"></a>
<a name="cl-2970"></a>
<a name="cl-2971"></a>    cdef int flat_index
<a name="cl-2972"></a>    cdef float outflow_weight
<a name="cl-2973"></a>    cdef float r_sum_avail
<a name="cl-2974"></a>    cdef float neighbor_r_sum_avail_pour
<a name="cl-2975"></a>    cdef float neighbor_sf_down
<a name="cl-2976"></a>    cdef float neighbor_sf
<a name="cl-2977"></a>    cdef float sf_down_sum
<a name="cl-2978"></a>    cdef float sf
<a name="cl-2979"></a>    cdef float r_avail
<a name="cl-2980"></a>    cdef float sf_down_frac
<a name="cl-2981"></a>    cdef int neighbor_direction
<a name="cl-2982"></a>
<a name="cl-2983"></a>
<a name="cl-2984"></a>    cdef time_t last_time, current_time
<a name="cl-2985"></a>    time(&amp;last_time)
<a name="cl-2986"></a>    LOGGER.info(
<a name="cl-2987"></a>                'START cells_to_process on SF route size: %d',
<a name="cl-2988"></a>                cells_to_process.size())
<a name="cl-2989"></a>    while cells_to_process.size() &gt; 0:
<a name="cl-2990"></a>        flat_index = cells_to_process.front()
<a name="cl-2991"></a>        cells_to_process.pop_front()
<a name="cl-2992"></a>        cells_in_queue.erase(flat_index)
<a name="cl-2993"></a>        global_row = flat_index / n_cols
<a name="cl-2994"></a>        global_col = flat_index % n_cols
<a name="cl-2995"></a>
<a name="cl-2996"></a>        block_cache.update_cache(
<a name="cl-2997"></a>            global_row, global_col, &amp;row_index, &amp;col_index,
<a name="cl-2998"></a>            &amp;row_block_offset, &amp;col_block_offset)
<a name="cl-2999"></a>
<a name="cl-3000"></a>        outflow_weight = outflow_weights_block[
<a name="cl-3001"></a>            row_index, col_index, row_block_offset, col_block_offset]
<a name="cl-3002"></a>
<a name="cl-3003"></a>        outflow_direction = outflow_direction_block[
<a name="cl-3004"></a>            row_index, col_index, row_block_offset, col_block_offset]
<a name="cl-3005"></a>        sf = sf_block[row_index, col_index, row_block_offset, col_block_offset]
<a name="cl-3006"></a>
<a name="cl-3007"></a>        time(&amp;current_time)
<a name="cl-3008"></a>        if current_time - last_time &gt; 5.0:
<a name="cl-3009"></a>            last_time = current_time
<a name="cl-3010"></a>            LOGGER.info(
<a name="cl-3011"></a>                'cells_to_process on SF route size: %d',
<a name="cl-3012"></a>                cells_to_process.size())
<a name="cl-3013"></a>            index_str = "[(%d, %d)," % (global_row, global_col)
<a name="cl-3014"></a>            dir_weight_str = "[(%d, %f, %f)," % (outflow_direction, outflow_weight, sf)
<a name="cl-3015"></a>            count = 8
<a name="cl-3016"></a>            for cell in cells_to_process:
<a name="cl-3017"></a>                count -= 1
<a name="cl-3018"></a>                cell_row = cell / n_cols
<a name="cl-3019"></a>                cell_col = cell % n_cols
<a name="cl-3020"></a>                index_str += "(%d, %d)," % (cell_row, cell_col)
<a name="cl-3021"></a>
<a name="cl-3022"></a>                block_cache.update_cache(
<a name="cl-3023"></a>                    cell_row, cell_col, &amp;row_index, &amp;col_index,
<a name="cl-3024"></a>                    &amp;row_block_offset, &amp;col_block_offset)
<a name="cl-3025"></a>
<a name="cl-3026"></a>                outflow_weight = outflow_weights_block[
<a name="cl-3027"></a>                    row_index, col_index, row_block_offset, col_block_offset]
<a name="cl-3028"></a>                outflow_direction = outflow_direction_block[
<a name="cl-3029"></a>                    row_index, col_index, row_block_offset, col_block_offset]
<a name="cl-3030"></a>                sf = sf_block[row_index, col_index, row_block_offset, col_block_offset]
<a name="cl-3031"></a>
<a name="cl-3032"></a>                dir_weight_str += "(%d, %f, %f)," % (outflow_direction, outflow_weight, sf)
<a name="cl-3033"></a>
<a name="cl-3034"></a>                if count == 0: break
<a name="cl-3035"></a>            index_str += '...]'
<a name="cl-3036"></a>            dir_weight_str += '...]'
<a name="cl-3037"></a>            LOGGER.debug(index_str)
<a name="cl-3038"></a>            LOGGER.debug(dir_weight_str)
<a name="cl-3039"></a>            block_cache.flush_cache()
<a name="cl-3040"></a>
<a name="cl-3041"></a>
<a name="cl-3042"></a>        block_cache.update_cache(
<a name="cl-3043"></a>            global_row, global_col, &amp;row_index, &amp;col_index,
<a name="cl-3044"></a>            &amp;row_block_offset, &amp;col_block_offset)
<a name="cl-3045"></a>        #if cell is processed, then skip
<a name="cl-3046"></a>        if sf_block[row_index, col_index, row_block_offset, col_block_offset] != sf_nodata:
<a name="cl-3047"></a>            continue
<a name="cl-3048"></a>
<a name="cl-3049"></a>        if outflow_direction == outflow_direction_nodata:
<a name="cl-3050"></a>            r_sum_avail = r_sum_avail_block[
<a name="cl-3051"></a>                row_index, col_index, row_block_offset, col_block_offset]
<a name="cl-3052"></a>            if r_sum_avail == r_sum_nodata:
<a name="cl-3053"></a>                sf_down_block[row_index, col_index, row_block_offset, col_block_offset] = 0.0
<a name="cl-3054"></a>                sf_block[row_index, col_index, row_block_offset, col_block_offset] = 0.0
<a name="cl-3055"></a>            else:
<a name="cl-3056"></a>                sf_down_sum = r_sum_avail / 1000.0 * pixel_area
<a name="cl-3057"></a>                sf_down_block[row_index, col_index, row_block_offset, col_block_offset] = sf_down_sum
<a name="cl-3058"></a>                r_avail = r_avail_block[row_index, col_index, row_block_offset, col_block_offset]
<a name="cl-3059"></a>                if r_sum_avail != 0:
<a name="cl-3060"></a>                    sf_block[row_index, col_index, row_block_offset, col_block_offset] = max(sf_down_sum * r_avail / (r_avail+r_sum_avail), 0)
<a name="cl-3061"></a>                else:
<a name="cl-3062"></a>                    sf_block[row_index, col_index, row_block_offset, col_block_offset] = 0.0
<a name="cl-3063"></a>            cache_dirty[row_index, col_index] = 1
<a name="cl-3064"></a>        elif stream_block[row_index, col_index, row_block_offset, col_block_offset] == 1:
<a name="cl-3065"></a>            sf_down_block[row_index, col_index, row_block_offset, col_block_offset] = 0.0
<a name="cl-3066"></a>            sf_block[row_index, col_index, row_block_offset, col_block_offset] = 0.0
<a name="cl-3067"></a>            cache_dirty[row_index, col_index] = 1
<a name="cl-3068"></a>        else:
<a name="cl-3069"></a>            downstream_calculated = 1
<a name="cl-3070"></a>            sf_down_sum = 0.0
<a name="cl-3071"></a>            for neighbor_index in xrange(2):
<a name="cl-3072"></a>                if neighbor_index == 1:
<a name="cl-3073"></a>                    outflow_direction = (outflow_direction + 1) % 8
<a name="cl-3074"></a>                    outflow_weight = 1.0 - outflow_weight
<a name="cl-3075"></a>
<a name="cl-3076"></a>                if outflow_weight &lt;= 0.0:
<a name="cl-3077"></a>                    #doesn't flow here, so skip
<a name="cl-3078"></a>                    continue
<a name="cl-3079"></a>
<a name="cl-3080"></a>                neighbor_row = global_row + row_offsets[outflow_direction]
<a name="cl-3081"></a>                neighbor_col = global_col + col_offsets[outflow_direction]
<a name="cl-3082"></a>                if (neighbor_row &lt; 0 or neighbor_row &gt;= n_rows or
<a name="cl-3083"></a>                        neighbor_col &lt; 0 or neighbor_col &gt;= n_cols):
<a name="cl-3084"></a>                    #out of bounds
<a name="cl-3085"></a>                    continue
<a name="cl-3086"></a>
<a name="cl-3087"></a>                block_cache.update_cache(
<a name="cl-3088"></a>                    neighbor_row, neighbor_col, &amp;neighbor_row_index,
<a name="cl-3089"></a>                    &amp;neighbor_col_index, &amp;neighbor_row_block_offset,
<a name="cl-3090"></a>                    &amp;neighbor_col_block_offset)
<a name="cl-3091"></a>
<a name="cl-3092"></a>                if stream_block[
<a name="cl-3093"></a>                        neighbor_row_index, neighbor_col_index,
<a name="cl-3094"></a>                        neighbor_row_block_offset, neighbor_col_block_offset] == 1:
<a name="cl-3095"></a>                    #calc base case
<a name="cl-3096"></a>                    r_sum_avail = r_sum_avail_block[
<a name="cl-3097"></a>                        row_index, col_index, row_block_offset, col_block_offset]
<a name="cl-3098"></a>                    sf_down_sum += outflow_weight * r_sum_avail / 1000.0 * pixel_area
<a name="cl-3099"></a>                else:
<a name="cl-3100"></a>                    if sf_block[neighbor_row_index, neighbor_col_index,
<a name="cl-3101"></a>                        neighbor_row_block_offset, neighbor_col_block_offset] == sf_nodata:
<a name="cl-3102"></a>                        #push neighbor on stack
<a name="cl-3103"></a>                        downstream_calculated = 0
<a name="cl-3104"></a>                        neighbor_flat_index = neighbor_row * n_cols + neighbor_col
<a name="cl-3105"></a>                        #push original on the end of the deque
<a name="cl-3106"></a>                        if (cells_in_queue.find(flat_index) ==
<a name="cl-3107"></a>                            cells_in_queue.end()):
<a name="cl-3108"></a>                            cells_to_process.push_back(flat_index)
<a name="cl-3109"></a>                            cells_in_queue.insert(flat_index)
<a name="cl-3110"></a>
<a name="cl-3111"></a>                        #push neighbor on front of deque
<a name="cl-3112"></a>                        if (cells_in_queue.find(neighbor_flat_index) ==
<a name="cl-3113"></a>                            cells_in_queue.end()):
<a name="cl-3114"></a>                            cells_to_process.push_front(neighbor_flat_index)
<a name="cl-3115"></a>                            cells_in_queue.insert(neighbor_flat_index)
<a name="cl-3116"></a>
<a name="cl-3117"></a>                    else:
<a name="cl-3118"></a>                        #calculate downstream contribution
<a name="cl-3119"></a>                        neighbor_r_sum_avail_pour = r_sum_avail_pour_block[
<a name="cl-3120"></a>                            neighbor_row_index, neighbor_col_index,
<a name="cl-3121"></a>                            neighbor_row_block_offset, neighbor_col_block_offset]
<a name="cl-3122"></a>                        if neighbor_r_sum_avail_pour != 0:
<a name="cl-3123"></a>                            neighbor_sf_down = sf_down_block[
<a name="cl-3124"></a>                                neighbor_row_index, neighbor_col_index,
<a name="cl-3125"></a>                                neighbor_row_block_offset, neighbor_col_block_offset]
<a name="cl-3126"></a>                            neighbor_sf = sf_block[
<a name="cl-3127"></a>                                neighbor_row_index, neighbor_col_index,
<a name="cl-3128"></a>                                neighbor_row_block_offset, neighbor_col_block_offset]
<a name="cl-3129"></a>                            block_cache.update_cache(
<a name="cl-3130"></a>                                global_row, global_col, &amp;row_index, &amp;col_index,
<a name="cl-3131"></a>                                &amp;row_block_offset, &amp;col_block_offset)
<a name="cl-3132"></a>                            r_sum_avail = r_sum_avail_block[
<a name="cl-3133"></a>                                row_index, col_index, row_block_offset, col_block_offset]
<a name="cl-3134"></a>                            if neighbor_sf &gt; neighbor_sf_down:
<a name="cl-3135"></a>                                LOGGER.error('%f, %f, %f, %f, %f', neighbor_sf,
<a name="cl-3136"></a>                                    neighbor_sf_down, r_sum_avail, neighbor_r_sum_avail_pour,
<a name="cl-3137"></a>                                    outflow_weight)
<a name="cl-3138"></a>                            sf_down_frac = outflow_weight * r_sum_avail / neighbor_r_sum_avail_pour
<a name="cl-3139"></a>                            if sf_down_frac &gt; 1.0: #can happen because of roundoff error
<a name="cl-3140"></a>                                sf_down_frac = 1.0
<a name="cl-3141"></a>                            sf_down_sum +=  outflow_weight * (neighbor_sf_down - neighbor_sf) * sf_down_frac
<a name="cl-3142"></a>                            if sf_down_sum &lt; 0:
<a name="cl-3143"></a>                                pass#LOGGER.error(sf_down_sum)
<a name="cl-3144"></a>
<a name="cl-3145"></a>            if downstream_calculated:
<a name="cl-3146"></a>                block_cache.update_cache(
<a name="cl-3147"></a>                    global_row, global_col, &amp;row_index, &amp;col_index,
<a name="cl-3148"></a>                    &amp;row_block_offset, &amp;col_block_offset)
<a name="cl-3149"></a>                #add contribution of neighbors to calculate si_down and si on current pixel
<a name="cl-3150"></a>                r_avail = r_avail_block[row_index, col_index, row_block_offset, col_block_offset]
<a name="cl-3151"></a>                r_sum_avail = r_sum_avail_block[row_index, col_index, row_block_offset, col_block_offset]
<a name="cl-3152"></a>                sf_down_block[row_index, col_index, row_block_offset, col_block_offset] = sf_down_sum
<a name="cl-3153"></a>                if r_sum_avail == 0:
<a name="cl-3154"></a>                    sf_block[row_index, col_index, row_block_offset, col_block_offset] = 0.0
<a name="cl-3155"></a>                else:
<a name="cl-3156"></a>                    #could r_sum_avail be &lt; than r_i in this case?
<a name="cl-3157"></a>                    sf_block[row_index, col_index, row_block_offset, col_block_offset] = max(sf_down_sum * r_avail / (r_avail+r_sum_avail), 0)
<a name="cl-3158"></a>                cache_dirty[row_index, col_index] = 1
<a name="cl-3159"></a>
<a name="cl-3160"></a>        #put upstream neighbors on stack for processing
<a name="cl-3161"></a>        for neighbor_index in xrange(8):
<a name="cl-3162"></a>            neighbor_row = neighbor_row_offset[neighbor_index] + global_row
<a name="cl-3163"></a>            neighbor_col = neighbor_col_offset[neighbor_index] + global_col
<a name="cl-3164"></a>
<a name="cl-3165"></a>            if (neighbor_row &gt;= n_rows or neighbor_row &lt; 0 or
<a name="cl-3166"></a>                    neighbor_col &gt;= n_cols or neighbor_col &lt; 0):
<a name="cl-3167"></a>                continue
<a name="cl-3168"></a>
<a name="cl-3169"></a>            block_cache.update_cache(
<a name="cl-3170"></a>                neighbor_row, neighbor_col,
<a name="cl-3171"></a>                &amp;neighbor_row_index, &amp;neighbor_col_index,
<a name="cl-3172"></a>                &amp;neighbor_row_block_offset,
<a name="cl-3173"></a>                &amp;neighbor_col_block_offset)
<a name="cl-3174"></a>
<a name="cl-3175"></a>            neighbor_direction = outflow_direction_block[
<a name="cl-3176"></a>                neighbor_row_index, neighbor_col_index,
<a name="cl-3177"></a>                neighbor_row_block_offset, neighbor_col_block_offset]
<a name="cl-3178"></a>            if neighbor_direction == outflow_direction_nodata:
<a name="cl-3179"></a>                continue
<a name="cl-3180"></a>
<a name="cl-3181"></a>            #check if the cell flows directly, or is one index off
<a name="cl-3182"></a>            if (inflow_offsets[neighbor_index] != neighbor_direction and
<a name="cl-3183"></a>                    ((inflow_offsets[neighbor_index] - 1) % 8) != neighbor_direction):
<a name="cl-3184"></a>                #then neighbor doesn't inflow into current cell
<a name="cl-3185"></a>                continue
<a name="cl-3186"></a>
<a name="cl-3187"></a>            #Calculate the outflow weight
<a name="cl-3188"></a>            outflow_weight = outflow_weights_block[
<a name="cl-3189"></a>                neighbor_row_index, neighbor_col_index,
<a name="cl-3190"></a>                neighbor_row_block_offset, neighbor_col_block_offset]
<a name="cl-3191"></a>
<a name="cl-3192"></a>            if ((inflow_offsets[neighbor_index] - 1) % 8) == neighbor_direction:
<a name="cl-3193"></a>                outflow_weight = 1.0 - outflow_weight
<a name="cl-3194"></a>
<a name="cl-3195"></a>            if outflow_weight &lt;= 0.0:
<a name="cl-3196"></a>                continue
<a name="cl-3197"></a>
<a name="cl-3198"></a>            #already processed, no need to loop on it again
<a name="cl-3199"></a>            if sf_block[neighbor_row_index, neighbor_col_index,
<a name="cl-3200"></a>                neighbor_row_block_offset, neighbor_col_block_offset] != sf_nodata:
<a name="cl-3201"></a>                continue
<a name="cl-3202"></a>
<a name="cl-3203"></a>            neighbor_flat_index = neighbor_row * n_cols + neighbor_col
<a name="cl-3204"></a>            if cells_in_queue.find(neighbor_flat_index) == cells_in_queue.end():
<a name="cl-3205"></a>                cells_to_process.push_back(neighbor_flat_index)
<a name="cl-3206"></a>                cells_in_queue.insert(neighbor_flat_index)
<a name="cl-3207"></a>
<a name="cl-3208"></a>        #if downstream aren't processed; skip and process those
<a name="cl-3209"></a>        #calc current pixel
<a name="cl-3210"></a>            #for each downstream neighbor
<a name="cl-3211"></a>                #if downstream pixel is a stream, then base case
<a name="cl-3212"></a>                #otherwise downstream case
<a name="cl-3213"></a>        #push upstream neighbors on for processing
<a name="cl-3214"></a>
<a name="cl-3215"></a>    block_cache.flush_cache()
<a name="cl-3216"></a>
</pre></div></td></tr></table>

  </div>


      </div>
    </div>
    <div data-modules="js/source/set-changeset" data-hash="0fbfb6cecdbdfb95b58f06d9fced7932fd9ec89b"></div>




  
  
    <script id="branch-dialog-template" type="text/html">
  

<div class="tabbed-filter-widget branch-dialog">
  <div class="tabbed-filter">
    <input placeholder="Filter branches" class="filter-box" autosave="branch-dropdown-10013696" type="text">
    [[^ignoreTags]]
      <div class="aui-tabs horizontal-tabs aui-tabs-disabled filter-tabs">
        <ul class="tabs-menu">
          <li class="menu-item active-tab"><a href="#branches">Branches</a></li>
          <li class="menu-item"><a href="#tags">Tags</a></li>
        </ul>
      </div>
    [[/ignoreTags]]
  </div>
  
    <div class="tab-pane active-pane" id="branches" data-filter-placeholder="Filter branches">
      <ol class="filter-list">
        <li class="empty-msg">No matching branches</li>
        [[#branches]]
          
            [[#hasMultipleHeads]]
              [[#heads]]
                <li class="comprev filter-item">
                  <a class="pjax-trigger filter-item-link" href="/natcap/invest/src/[[changeset]]/src/natcap/invest/seasonal_water_yield/seasonal_water_yield_core.pyx?at=[[safeName]]"
                     title="[[name]]">
                    [[name]] ([[shortChangeset]])
                  </a>
                </li>
              [[/heads]]
            [[/hasMultipleHeads]]
            [[^hasMultipleHeads]]
              <li class="comprev filter-item">
                <a class="pjax-trigger filter-item-link" href="/natcap/invest/src/[[changeset]]/src/natcap/invest/seasonal_water_yield/seasonal_water_yield_core.pyx?at=[[safeName]]" title="[[name]]">
                  [[name]]
                </a>
              </li>
            [[/hasMultipleHeads]]
          
        [[/branches]]
      </ol>
    </div>
    <div class="tab-pane" id="tags" data-filter-placeholder="Filter tags">
      <ol class="filter-list">
        <li class="empty-msg">No matching tags</li>
        [[#tags]]
          <li class="comprev filter-item">
            <a class="pjax-trigger filter-item-link" href="/natcap/invest/src/[[changeset]]/src/natcap/invest/seasonal_water_yield/seasonal_water_yield_core.pyx?at=[[safeName]]" title="[[name]]">
              [[name]]
            </a>
          </li>
        [[/tags]]
      </ol>
    </div>
  
</div>

</script>
  



  </div>

        
        
        
      </div>
    </div>
  </div>

    </div>
  </div>

  <footer id="footer" role="contentinfo" data-modules="components/footer">
    <section class="footer-body">
      
  <ul>
  <li>
    <a class="support-ga" target="_blank"
       data-support-gaq-page="Blog"
       href="http://blog.bitbucket.org">Blog</a>
  </li>
  <li>
    <a class="support-ga" target="_blank"
       data-support-gaq-page="Home"
       href="/support">Support</a>
  </li>
  <li>
    <a class="support-ga"
       data-support-gaq-page="PlansPricing"
       href="/plans">Plans &amp; pricing</a>
  </li>
  <li>
    <a class="support-ga" target="_blank"
       data-support-gaq-page="DocumentationHome"
       href="//confluence.atlassian.com/display/BITBUCKET">Documentation</a>
  </li>
  <li>
    <a class="support-ga" target="_blank"
       data-support-gaq-page="DocumentationAPI"
       href="//confluence.atlassian.com/x/IYBGDQ">API</a>
  </li>
  <li>
    <a class="support-ga" target="_blank"
       data-support-gaq-page="SiteStatus"
       href="http://status.bitbucket.org/">Site status</a>
  </li>
  <li>
    <a class="support-ga" id="meta-info"
       data-support-gaq-page="MetaInfo"
       href="#">Version info</a>
  </li>
  <li>
    <a class="support-ga" target="_blank"
       data-support-gaq-page="EndUserAgreement"
       href="//www.atlassian.com/end-user-agreement?utm_source=bitbucket&amp;utm_medium=link&amp;utm_campaign=footer">Terms of service</a>
  </li>
  <li>
    <a class="support-ga" target="_blank"
       data-support-gaq-page="PrivacyPolicy"
       href="//www.atlassian.com/company/privacy?utm_source=bitbucket&amp;utm_medium=link&amp;utm_campaign=footer">Privacy policy</a>
  </li>
</ul>
<div id="meta-info-content" style="display: none;">
  <ul>
    
      <li>English</li>
    
    <li>
      <a class="support-ga" target="_blank"
         data-support-gaq-page="GitDocumentation"
         href="http://git-scm.com/">Git 2.1.1</a>
    </li>
    <li>
      <a class="support-ga" target="_blank"
         data-support-gaq-page="HgDocumentation"
         href="http://mercurial.selenic.com/">Mercurial 2.9</a>
    </li>
    <li>
      <a class="support-ga" target="_blank"
         data-support-gaq-page="DjangoDocumentation"
         href="https://www.djangoproject.com/">Django 1.7.8</a>
    </li>
    <li>
      <a class="support-ga" target="_blank"
         data-support-gaq-page="PythonDocumentation"
         href="http://www.python.org/">Python 2.7.3</a>
    </li>
    <li>
      <a class="support-ga" target="_blank"
         data-support-gaq-page="DeployedVersion"
         href="#">dc76a8f1e35f / 4a1a7b368c96 @ app19</a>
    </li>
  </ul>
</div>
<ul class="atlassian-links">
  <li>
    <a id="atlassian-jira-link" target="_blank"
       title="Track everything – bugs, tasks, deadlines, code – and pull reports to stay informed."
       href="http://www.atlassian.com/software/jira?utm_source=bitbucket&amp;utm_medium=link&amp;utm_campaign=bitbucket_footer">JIRA</a>
  </li>
  <li>
    <a id="atlassian-confluence-link" target="_blank"
       title="Content Creation, Collaboration & Knowledge Sharing for Teams."
       href="http://www.atlassian.com/software/confluence/overview/team-collaboration-software?utm_source=bitbucket&amp;utm_medium=link&amp;utm_campaign=bitbucket_footer">Confluence</a>
  </li>
  <li>
    <a id="atlassian-bamboo-link" target="_blank"
       title="Continuous integration and deployment, release management."
       href="http://www.atlassian.com/software/bamboo/overview?utm_source=bitbucket&amp;utm_medium=link&amp;utm_campaign=bitbucket_footer">Bamboo</a>
  </li>
  <li>
    <a id="atlassian-stash-link" target="_blank"
       title="Git repo management, behind your firewall and Enterprise-ready."
       href="http://www.atlassian.com/software/stash?utm_source=bitbucket&amp;utm_medium=link&amp;utm_campaign=bitbucket_footer">Stash</a>
  </li>
  <li>
    <a id="atlassian-sourcetree-link" target="_blank"
       title="A free Git and Mercurial desktop client for Mac or Windows."
       href="http://www.sourcetreeapp.com/?utm_source=bitbucket&amp;utm_medium=link&amp;utm_campaign=bitbucket_footer">SourceTree</a>
  </li>
  <li>
    <a id="atlassian-hipchat-link" target="_blank"
       title="Group chat and IM built for teams."
       href="http://www.hipchat.com/?utm_source=bitbucket&amp;utm_medium=link&amp;utm_campaign=bitbucket_footer">HipChat</a>
  </li>
</ul>
<div id="footer-logo">
  <a target="_blank"
     title="Bitbucket is developed by Atlassian in San Francisco and Austin."
     href="http://www.atlassian.com?utm_source=bitbucket&amp;utm_medium=logo&amp;utm_campaign=bitbucket_footer">Atlassian</a>
</div>

    </section>
  </footer>
</div>


  

<script src="https://d3oaxc4q5k2d6q.cloudfront.net/m/4a1a7b368c96/jsi18n/en/djangojs.js"></script>

  
    <script src="https://d3oaxc4q5k2d6q.cloudfront.net/m/4a1a7b368c96/dist/main.js" data-main="https://d3oaxc4q5k2d6q.cloudfront.net/m/4a1a7b368c96/dist/main" defer></script>
  



<script>
  (function () {
    var ga = document.createElement('script');
    ga.src = ('https:' == document.location.protocol ? 'https://ssl' : 'http://www') + '.google-analytics.com/ga.js';
    ga.setAttribute('async', 'true');
    var s = document.getElementsByTagName('script')[0];
    s.parentNode.insertBefore(ga, s);
  }());
</script>


  

<div data-modules="components/mentions/index">
  <script id="mention-result" type="text/html">
    
<div class="aui-avatar aui-avatar-small">
  <div class="aui-avatar-inner">
    <img src="[[avatar_url]]">
  </div>
</div>
[[#display_name]]
  <span class="display-name">[[&display_name]]</span> <small class="username">[[&username]]</small>
[[/display_name]]
[[^display_name]]
  <span class="username">[[&username]]</span>
[[/display_name]]
[[#is_teammate]][[^is_team]]
  <span class="aui-lozenge aui-lozenge-complete aui-lozenge-subtle">teammate</span>
[[/is_team]][[/is_teammate]]

  </script>
  <script id="mention-call-to-action" type="text/html">
    
[[^query]]
<li class="bb-typeahead-item">Begin typing to search for a user</li>
[[/query]]
[[#query]]
<li class="bb-typeahead-item">Continue typing to search for a user</li>
[[/query]]

  </script>
  <script id="mention-no-results" type="text/html">
    
[[^searching]]
<li class="bb-typeahead-item">Found no matching users for <em>[[query]]</em>.</li>
[[/searching]]
[[#searching]]
<li class="bb-typeahead-item bb-typeahead-searching">Searching for <em>[[query]]</em>.</li>
[[/searching]]

  </script>
</div>

  <div data-modules="components/typeahead/emoji/index">
    <script id="emoji-result" type="text/html">
    
<div class="aui-avatar aui-avatar-small">
  <div class="aui-avatar-inner">
    <img src="[[src]]">
  </div>
</div>
<span class="name">[[&name]]</span>

  </script>
  </div>


<div data-modules="components/repo-typeahead/index">
  <script id="repo-typeahead-result" type="text/html">
    <span class="aui-avatar aui-avatar-project aui-avatar-xsmall">
  <span class="aui-avatar-inner">
    <img src="[[avatar]]">
  </span>
</span>
<span class="owner">[[&owner]]</span>/<span class="slug">[[&slug]]</span>

  </script>
</div>
<script id="share-form-template" type="text/html">
    

<div class="error aui-message hidden">
  <span class="aui-icon icon-error"></span>
  <div class="message"></div>
</div>
<form class="aui">
  <table class="widget bb-list aui">
    <thead>
    <tr class="assistive">
      <th class="user">User</th>
      <th class="role">Role</th>
      <th class="actions">Actions</th>
    </tr>
    </thead>
    <tbody>
      <tr class="form">
        <td colspan="2">
          <input type="text" class="text bb-user-typeahead user-or-email"
                 placeholder="Username or email address"
                 autocomplete="off"
                 data-bb-typeahead-focus="false"
                 [[#disabled]]disabled[[/disabled]]>
        </td>
        <td class="actions">
          <button type="submit" class="aui-button" disabled>Add</button>
        </td>
      </tr>
    </tbody>
  </table>
</form>

  </script>
<script id="share-detail-template" type="text/html">
    

[[#username]]
<td class="user
           [[#hasCustomGroups]]custom-groups[[/hasCustomGroups]]"
    [[#error]]data-error="[[error]]"[[/error]]>
  <div title="[[displayName]]">
    <a href="/[[username]]" class="user">
      <img class="avatar avatar16" src="[[avatar]]" />
      <span>[[displayName]]</span>
    </a>
  </div>
</td>
[[/username]]
[[^username]]
<td class="email
           [[#hasCustomGroups]]custom-groups[[/hasCustomGroups]]"
    [[#error]]data-error="[[error]]"[[/error]]>
  <div title="[[email]]">
    <span class="aui-icon aui-icon-small aui-iconfont-email"></span>
    [[email]]
  </div>
</td>
[[/username]]
<td class="role
           [[#hasCustomGroups]]custom-groups[[/hasCustomGroups]]">
  <div>
    [[#group]]
      [[#hasCustomGroups]]
        <select class="group [[#noGroupChoices]]hidden[[/noGroupChoices]]">
          [[#groups]]
            <option value="[[slug]]"
                    [[#isSelected]]selected[[/isSelected]]>
              [[name]]
            </option>
          [[/groups]]
        </select>
      [[/hasCustomGroups]]
      [[^hasCustomGroups]]
      <label>
        <input type="checkbox" class="admin"
               [[#isAdmin]]checked[[/isAdmin]]>
        Administrator
      </label>
      [[/hasCustomGroups]]
    [[/group]]
    [[^group]]
      <ul>
        <li class="permission aui-lozenge aui-lozenge-complete
                   [[^read]]aui-lozenge-subtle[[/read]]"
            data-permission="read">
          read
        </li>
        <li class="permission aui-lozenge aui-lozenge-complete
                   [[^write]]aui-lozenge-subtle[[/write]]"
            data-permission="write">
          write
        </li>
        <li class="permission aui-lozenge aui-lozenge-complete
                   [[^admin]]aui-lozenge-subtle[[/admin]]"
            data-permission="admin">
          admin
        </li>
      </ul>
    [[/group]]
  </div>
</td>
<td class="actions
           [[#hasCustomGroups]]custom-groups[[/hasCustomGroups]]">
  <div>
    <a href="#" class="delete">
      <span class="aui-icon aui-icon-small aui-iconfont-remove">Delete</span>
    </a>
  </div>
</td>

  </script>
<script id="share-team-template" type="text/html">
    

<div class="clearfix">
  <span class="team-avatar-container">
    <img src="[[avatar]]" alt="[[display_name]]" title="[[display_name]]" class="avatar avatar32" />
  </span>
  <span class="team-name-container">
    [[display_name]]
  </span>
</div>
<p class="helptext">
  
    Existing users are granted access to this team immediately.
    New users will be sent an invitation.
  
</p>
<div class="share-form"></div>

  </script>


  

<script id="source-changeset" type="text/html">
  

<a href="/natcap/invest/src/[[raw_node]]/[[path]]?at=develop"
   class="[[#selected]]highlight[[/selected]]"
   data-hash="[[node]]">
  [[#author.username]]
    <img class="avatar avatar16" src="[[author.avatar]]"/>
    <span class="author" title="[[raw_author]]">[[author.display_name]]</span>
  [[/author.username]]
  [[^author.username]]
    <img class="avatar avatar16" src="https://d3oaxc4q5k2d6q.cloudfront.net/m/4a1a7b368c96/img/default_avatar/16/user_blue.png"/>
    <span class="author unmapped" title="[[raw_author]]">[[author]]</span>
  [[/author.username]]
  <time datetime="[[utctimestamp]]" data-title="true">[[utctimestamp]]</time>
  <span class="message">[[message]]</span>
</a>

</script>
<script id="embed-template" type="text/html">
  

<form class="aui embed">
  <label for="embed-code">Embed this source in another page:</label>
  <input type="text" readonly="true" value="&lt;script src=&quot;[[url]]&quot;&gt;&lt;/script&gt;" id="embed-code">
</form>

</script>




  
  
  





<aui-inline-dialog2

      id="super-touch-point-dialog"

    
      class="aui-layer aui-inline-dialog"
      data-aui-alignment="bottom right"
    

    
    data-aui-alignment-static="true"
    data-aui-responds-to="toggle"
    data-modules="header/help-menu,js/connect/connect-views,js/connect/super-touch-point"
    aria-hidden="true">

  
  <div class="aui-inline-dialog-contents">
  

    <div id="ace-stp-section" class="no-touch-point">
      <div id="ace-stp-help-section">
        <h1 class="ace-stp-heading">Help</h1>

        <form id="ace-stp-search-form" class="aui" target="_blank" method="get"
            action="https://support.atlassian.com/customer/search">
          <span id="stp-search-icon" class="aui-icon aui-icon-large aui-iconfont-search"></span>
          <input id="ace-stp-search-form-input" name="q" class="text" type="text" placeholder="Ask a question">
        </form>

        <ul id="ace-stp-help-links">
          <li>
            <a class="support-ga" data-support-gaq-page="DocumentationHome"
                href="https://confluence.atlassian.com/x/bgozDQ" target="_blank">
              Online help
            </a>
          </li>
          <li>
            <a class="support-ga" data-support-gaq-page="GitTutorials"
                href="https://www.atlassian.com/git?utm_source=bitbucket&amp;utm_medium=link&amp;utm_campaign=help_dropdown&amp;utm_content=learn_git"
                target="_blank">
              Learn Git
            </a>
          </li>
          <li>
            <a id="keyboard-shortcuts-link"
               href="#">Keyboard shortcuts</a>
          </li>
          <li>
            <a href="/whats-new" id="features-link">
              Latest features
            </a>
          </li>
          <li>
            <a class="support-ga" data-support-gaq-page="Documentation101"
                href="https://confluence.atlassian.com/x/cgozDQ" target="_blank">
              Bitbucket 101
            </a>
          </li>
          <li>
            <a class="support-ga" data-support-gaq-page="SiteStatus"
                href="http://status.bitbucket.org/" target="_blank">
              Site status
            </a>
          </li>
          <li>
            <a class="support-ga" data-support-gaq-page="Home" href="/support">
              Support
            </a>
          </li>

        </ul>
      </div>

      
    </div>

  
  </div>
  


</aui-inline-dialog2>

  





</body>
</html>
