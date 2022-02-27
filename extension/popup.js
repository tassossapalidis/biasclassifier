
let runBiasClassifier = document.getElementById("runBiasClassifier");
let div = document.getElementById('showText');

// When the button is clicked, inject setPageBackgroundColor into current page
runBiasClassifier.addEventListener("click", async () => {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {

        let url = tabs[0].url;
        removeAllChildNodes(div);
        let url_child = document.createTextNode(url);
        div.appendChild(url_child);
    });
});

console.log('hello')

// helper function to remove nodes from div
function removeAllChildNodes(parent) {
    while (parent.firstChild) {
        parent.removeChild(parent.firstChild);
    }
}