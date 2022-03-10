let runPolarity = document.getElementById("runPolarity");
let divIsRun = document.getElementById('isRun');
let divResults1 = document.getElementById('result1');
let divResults2 = document.getElementById('result2');
let isloaded = false;

// url to Cloud Function
let api_url = 'https://us-west1-cs-329-bias-classifier.cloudfunctions.net/polarity_func-2';

// configure the gauge visual
var gaugeOpts = {
    // color configs
    colorStart: "#6fadcf",
    colorStop: void 0,
    gradientType: 0,
    strokeColor: "#e0e0e0",
    generateGradient: true,
    percentColors: [[0.0, "#a9d70b" ], [0.50, "#f9c802"], [1.0, "#ff0000"]],
    // customize pointer
    pointer: {
      length: 0.8,
      strokeWidth: 0.035,
      iconScale: 1.0
    },
    // static zones
    staticZones: [
      {strokeStyle: "#2E73E6", min: 0, max: .1},
      {strokeStyle: "#3BA8E7", min: .1, max: .25},
      {strokeStyle: "#FDE25D", min: .25, max: .75},
      {strokeStyle: "#FB607E", min: .75, max: .9},
      {strokeStyle: "#E92A39", min: .9, max: 1}
    ],
    // the span of the gauge arc
    angle: 0.15,
    // line thickness
    lineWidth: 0.44,
    // radius scale
    radiusScale: 1.0,
    // font size
    fontSize: 40,
    // if false, max value increases automatically if value > maxValue
    limitMax: false,
    // if true, the min value of the gauge will be fixed
    limitMin: false,
    // High resolution support
    highDpiSupport: true
};

// wait for click
document.addEventListener("DOMContentLoaded", function() {
    isloaded = true;
})

runPolarity.addEventListener("click", async() => {
    if (isloaded) {
        chrome.tabs.query({active: true, currentWindow: true}, async function(tabs) {
            amendHTML(tabs);
            let res = await getPrediction(tabs[0]);
            runPolarity.classList.remove("button--loading")
            runPolarity.textContent = `Complete`
    });
    }
});


// edit popup after button is clicked
function amendHTML(tabs) {
    runPolarity.textContent = ``;
    runPolarity.classList.add("button--loading");
    removeAllChildNodes(divResults1)

    let title = tabs[0].title;
    bold = document.createElement('strong')
    let title_child = document.createTextNode(title);

    removeAllChildNodes(divIsRun);
    divIsRun.appendChild(title_child);
}

// helper function to remove nodes from divIsRun
function removeAllChildNodes(parent) {
    while (parent.firstChild) {
        parent.removeChild(parent.firstChild);
    }
}

// function to request model outputs
async function getPrediction(tab) {
    JSON.stringify(tab.url);
    let response = await fetch(api_url, {
        method: 'POST',
        body: JSON.stringify({"url": tab.url}),
        headers: {
            "Content-Type": "application/json"
        }
    })
    .then(response => {return response.json()})
    .then(data => {
        let median_conf = data['med_conf'];
        let max_conf = data['max_conf'];
        let median_class = data['med_class'];
        let maximum_class = data['max_class'];
        let median_leaning = "liberal";
        let maximum_leaning = "liberal";
        if (median_class == 1) median_leaning = "conservative"
        if (maximum_class == 1) maximum_leaning = "conservative"

        let bias_display = '';
        if (median_leaning == 'conservative') {
            if (median_conf >= .9) {
                bias_display = 'is very conservative.'
            }
            else if (median_conf >= .75) {
                bias_display = 'leans conservative.'
            }
            else {
                bias_display = 'is politically neutral.'
            }
        }
        else {
            if (median_conf >= .9) {
                bias_display = 'is very liberal.'
            }
            else if (median_conf >= .75) {
                bias_display = 'leans liberal.'
            }
            else {
                bias_display = 'is politically neutral.'
            }
        }
        
        // configure div to hold predicted class
        let median_child = document.createTextNode('This article '.concat(bias_display));
        removeAllChildNodes(divResults1);
        divResults1.appendChild(median_child)

        let attr_sentences = data['bias_sentences']
        let substring = attr_sentences[0].substring(45, Math.min(95, attr_sentences[0].length))
        let splits = substring.split(".")
        let selectors = [];
        
        // script to send content to highlight to content.js
        for (let i = 0; i < splits.length; i++) {
            if (selectors.length > 0) {
                if (selectors[0].length < splits[i].length) {
                    selectors[0] = 'p:contains("'.concat(splits[i]).concat('")');
                }
            }
            else {
                selectors[0] = 'p:contains("'.concat(splits[i]).concat('")');
            }
        }

        let msg = {
            sels : selectors,
            bias_type : maximum_leaning,
            max_conf : max_conf,
        }

        chrome.tabs.sendMessage(tab.id,msg);

        // set up the gauge
        var target = document.getElementById('gauge'); 
        var gauge = new Gauge(target).setOptions(gaugeOpts);

        gauge.maxValue = 1;
        gauge.setMinValue(0); 
        let gauge_value = 0.5;
        if (median_leaning == 'conservative') {
            gauge_value = median_conf
        }
        else {
            gauge_value = 1 - median_conf
        }
        gauge.set(gauge_value);
        gauge.animationSpeed = 32
    })
    .catch(error => console.error(error));

    return response

}
