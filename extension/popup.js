let runPolarity = document.getElementById("runPolarity");
let divIsRun = document.getElementById('isRun');
let divResults1 = document.getElementById('result1');
let divResults2 = document.getElementById('result2');
let isloaded = false;

let api_url = 'https://us-west1-cs-329-bias-classifier.cloudfunctions.net/polarity_func-2';

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



function amendHTML(tabs) {
    runPolarity.textContent = ``;
    runPolarity.classList.add("button--loading");
    removeAllChildNodes(divResults1)
    removeAllChildNodes(divResults2);

    let title = tabs[0].title;
    //let results_child = document.createTextNode('Model results here');
    let title_child = document.createTextNode('Article title: '.concat(title));
    //divResults1.appendChild(results_child);

    removeAllChildNodes(divIsRun);
    divIsRun.appendChild(title_child);
}

// helper function to remove nodes from divIsRun
function removeAllChildNodes(parent) {
    while (parent.firstChild) {
        console.log(2);
        parent.removeChild(parent.firstChild);
    }
}

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
        console.log(data)
        let median_conf = data['med_conf'];
        let max_conf = data['max_conf'];
        let median_class = data['med_class'];
        let maximum_class = data['max_class'];
        let median_leaning = "liberal";
        let maximum_leaning = "liberal";
        if (median_class == 1) median_leaning = "conservative"
        if (maximum_class == 1) maximum_leaning = "conservative"

        let median_child = document.createTextNode('This article\'s median bias is '.concat(Math.round((median_conf * 100)).toString()).concat('% ').concat(median_leaning));
        let maximum_child = document.createTextNode('This article\'s maximum bias is '.concat(Math.round((max_conf * 100)).toString()).concat('% ').concat(maximum_leaning));

        removeAllChildNodes(divResults1);
        removeAllChildNodes(divResults2);
        divResults1.appendChild(median_child)
        divResults2.appendChild(maximum_child);

        let attr_sentences = data['bias_sentences']
        let substring = attr_sentences[0].substring(45, Math.min(95, attr_sentences[0].length))
        let splits = substring.split(".")
        let selectors = [];
        
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

        console.log(selectors)

        let msg = {
            sels : selectors,
            bias_type : maximum_leaning,
            max_conf : max_conf,
        }

        console.log(tab.id)
        chrome.tabs.sendMessage(tab.id,msg);

    })
    .catch(error => console.error(error));

    return response

}

