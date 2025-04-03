// Function to open the sidebar
function toggleNav() {
    var nav = document.getElementById("myNav");
    var navOptions = document.querySelector(".nav-options");
    if (nav.style.width === "250px") {
        nav.style.width = "0";
        navOptions.style.display = "none";
        document.body.classList.remove('sidenav-open');
    } else {
        nav.style.width = "250px";
        navOptions.style.display = "flex";
        document.body.classList.add('sidenav-open');
    }
}

// Function to close the sidebar
function closeNav() {
    var nav = document.getElementById("myNav");
    var navOptions = document.querySelector(".nav-options");
    nav.style.width = "0";
    navOptions.style.display = "none";
    document.body.classList.remove('sidenav-open');
}
