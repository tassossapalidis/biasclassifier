let runPolarity = document.getElementById("runPolarity");
let divIsRun = document.getElementById('isRun');
let divResults = document.getElementById('results');

let api_url = 'https://us-west1-cs-329-bias-classifier.cloudfunctions.net/polarity_func-2';

// When the button is clicked, inject setPageBackgroundColor into current page
runPolarity.addEventListener("click", async () => {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        if (runPolarity.textContent == `Complete`) {
            let run_child = document.createTextNode('Analysis already run on this page');
            removeAllChildNodes(divIsRun);
            divIsRun.appendChild(run_child);
        }
        else {
            let res = Promise.all([amendHTML(tabs), getPrediction(tabs[0].url)]);
            runPolarity.textContent = `Complete`;
            runPolarity.classList.remove("button--loading")
        }
    });
});

function amendHTML(tabs) {
    runPolarity.textContent = ``;
    runPolarity.classList.add("button--loading");
    let title = tabs[0].title;
    let results_child = document.createTextNode('Model results here');
    let title_child = document.createTextNode('Article title: '.concat(title));
    divResults.appendChild(results_child);
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

async function getPrediction(url) {
    JSON.stringify(url);
    let response = await fetch(api_url, {
        method: 'POST',
        body: JSON.stringify({"url": url}),
        headers: {
            "Content-Type": "application/json"
        }
    })
    .then(response => {return response.json()})
    .then(data => {
        let score = data[0][0];
        let score_child = document.createTextNode('This article is classified as '.concat(Math.round((score * 100)).toString()).concat('% liberal'));
        console.log(score);
        removeAllChildNodes(divResults);
        divResults.appendChild(score_child);
    })
    .catch(error => console.error(error));

    return response

    // .catch(error => alert('Error:', error));
    // .then(res => {
    //     $.each(res, function( index, value ) {
    //         let value_child = document.createTextNode(value);
    //         removeAllChildNodes(divResults);
    //         divResults.appendChild(value_child)
    //     });
    // })
    // .catch(error => console.error('Error:', error));
}



// // capture all text
// var textToSend = document.body.innerText;

// // summarize and send back
// const api_url = 'YOUR_GOOGLE_CLOUD_FUNCTION_URL'; 
// function getPrediction(url) {
//     fetch(api_url, {
//         method: 'POST',
//         body: JSON.stringify(textToSend),
//         headers:{
//           'Content-Type': 'application/json'
//         } })
//       .then(data => { return data.json() })
//       .then(res => { 
//           $.each(res, function( index, value ) {
//               value = unicodeToChar(value).replace(/\\n/g, '');
//               document.body.innerHTML = document.body.innerHTML.split(value).join('<span style="background-color: #fff799;">' + value + '</span>');
//           });
//        })
//       .catch(error => console.error('Error:', error));
// }

// // make REST API call
// function getPrediction(url) {
//     let instance = {"instances": [url]}
//     httpRequest = new XMLHttpRequest();
//     if (!httpRequest) {
//         alert('Giving up :( Cannot create an XMLHTTP instance');

//         return false;
//     }
//       httpRequest.onreadystatechange = alertContents(httpRequest);
//       httpRequest.open('POST', 'https://https://us-central1-ml.googleapis.com/v1/{name=projects/**}:predict');
//       httpRequest.send();
// }

// function alertContents(httpRequest) {
//     if (httpRequest.readyState === XMLHttpRequest.DONE) {
//       if (httpRequest.status === 200) {
//         alert(httpRequest.responseText);
//       } else {
//         alert('There was a problem with the request.');
//       }
//     }
// }