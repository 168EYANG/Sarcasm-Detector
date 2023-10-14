// // content.js
// function addSymbolToComments() {
//     document.addEventListener('DOMContentLoaded', function() {
//         const comments = document.querySelectorAll('.Comment');
    
//         comments.forEach(comment => {
//             const symbol = document.createElement('span');
//             symbol.textContent = '👉'; // You can change this symbol as needed
//             symbol.style.marginRight = '5px'; // Adjust spacing if necessary
//             comment.prepend(symbol);
//           });   
//     });
// }

// addSymbolToComments();

const comments = document.querySelectorAll('[data-testid=comment]');

comments.forEach(comment => {
    const symbol = document.createElement('span');
    if (comment.innerHTML.includes('True')) {
        symbol.textContent = '/srs';
    } else {
        symbol.textContent = '/s';
    }
    // symbol.textContent = '/s'; // You can change this symbol as needed
    symbol.style.marginRight = '5px'; // Adjust spacing if necessary
    comment.prepend(symbol);
});



