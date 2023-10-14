// const tweetContent = document.querySelectorAll('[data-testid=tweetText]');

const tweetContent = document.getElementsByClassName("css-901oao css-16my406 r-poiln3 r-bcqeeo r-qvutc0");

console.log(typeof tweetContent);

tweetContent.forEach(comment => {
    comment.style.display = "none";
});

// tweetContent.forEach(comment => {
//     const symbol = document.createElement('span');
//     if (comment.innerHTML.includes('tenz')) {
//         symbol.textContent = '/s';
//     } else {
//         symbol.textContent = '/srs';
//     }
//     // symbol.textContent = '/s'; // You can change this symbol as needed
//     symbol.style.marginRight = '5px'; // Adjust spacing if necessary
//     comment.prepend(symbol);
// });
