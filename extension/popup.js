let runPolarity = document.getElementById("runPolarity");
let divIsRun = document.getElementById('isRun');
let divResults = document.getElementById('results');

// When the button is clicked, inject setPageBackgroundColor into current page
runPolarity.addEventListener("click", async () => {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        if (runPolarity.textContent == `Complete`) {
            let run_child = document.createTextNode('Analysis already run on this page');
            removeAllChildNodes(divIsRun);
            divIsRun.appendChild(run_child);
        }
        else {
            runPolarity.textContent = ``
            // runPolarity.classList.add("button--loading");
            let title = tabs[0].title;
            let results_child = document.createTextNode('Model results here');
            let title_child = document.createTextNode('Article title: '.concat(title));
            divResults.appendChild(results_child);
            removeAllChildNodes(divIsRun);
            divIsRun.appendChild(title_child);
            // runPolarity.classList.remove("button--loading");
            runPolarity.textContent = `Complete`;
        }
    });
});

// helper function to remove nodes from divIsRun
function removeAllChildNodes(parent) {
    while (parent.firstChild) {
        parent.removeChild(parent.firstChild);
    }
}