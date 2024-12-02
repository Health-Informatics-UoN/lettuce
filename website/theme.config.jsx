export default {
  logo: <span>Lettuce documentation</span>,
  project: {
    link: 'https://github.com/health-informatics-uon/lettuce'
  },
  footer: {
    content: (
    <span>
      © {new Date().getFullYear()}{" "}
        <a href="https://nottingham.ac.uk" target="_blank">
          <img
            src="/lettuce/uon_white_text_web.png"
            alt="University of Nottingham"
            width={243}
            height={90}
          />
        </a>
        <br />
        MIT licence
    </span>
    )
  }
}
