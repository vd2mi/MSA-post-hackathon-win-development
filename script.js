// MSA Core Scripts
// Handles global interactions and keyboard shortcuts for all pages

document.addEventListener('DOMContentLoaded', () => {
    console.log('MSA Core Systems: ONLINE 🚀');

    // --- Global Keyboard Shortcuts ---
    document.addEventListener('keydown', (e) => {
        

        const inputs = {
            hero: document.getElementById('hero-ticker'),      // index.html
            dash: document.getElementById('ticker-input'),     // dashboard.html
            compA: document.getElementById('ticker-a'),        // compare.html (Input A)
            compB: document.getElementById('ticker-b')         // compare.html (Input B)
        };

        let targetInput = inputs.dash || inputs.hero;

        if (inputs.compA && inputs.compB) {

            if (inputs.compA.value.trim() !== '' && inputs.compB.value.trim() === '') {
                targetInput = inputs.compB;
            } else {

                targetInput = inputs.compA;
            }
        }

        if (!targetInput) return;


        const activeTag = document.activeElement.tagName;
        if (e.key === '/' && activeTag !== 'INPUT' && activeTag !== 'TEXTAREA') {
            e.preventDefault(); 
            targetInput.focus();

            if (targetInput === inputs.hero || targetInput === inputs.compA || targetInput === inputs.compB) {

                targetInput.parentElement.classList.add('ring-2', 'ring-bull', 'ring-opacity-50');
                setTimeout(() => targetInput.parentElement.classList.remove('ring-2', 'ring-bull', 'ring-opacity-50'), 200);
            } 
            else if (targetInput === inputs.dash) {
                
                targetInput.classList.add('bg-bull/10');
                setTimeout(() => targetInput.classList.remove('bg-bull/10'), 200);
            }
        }

        // --- Shortcut 2: Press "Esc" to Blur (Unfocus) ---
        if (e.key === 'Escape') {
            document.activeElement.blur();
        }
    });
});